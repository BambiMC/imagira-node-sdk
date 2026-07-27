# Host integration — adopting the SDK without forking the contract

For maintainers of Imagira (or any host app that runs these nodes), not for node
authors.

The risk isn't writing the SDK; it's that the main app has 100+ built-in nodes
importing `core.nodes.base_node` directly, plus the discovery, registry, lint
and pipeline machinery built around that module. If the SDK becomes a *second*
`BaseNode`, you get two registries and `isinstance` checks that silently fail
across the boundary.

**The rule: the app depends on the SDK and re-exports from it. It never keeps
its own copy.**

Reviewed against `imagira-node-sdk` 1.1.0 and the Imagira app at commit
`5ba1b92` + working tree, 2026-07-25. This is the canonical copy; the app repo
keeps a working duplicate at its root.

---

## Step 0 — the extraction drifted once already. Resolved in SDK 1.1.0.

The doc used to open by asking you to *confirm* the extraction was still
faithful. It wasn't — and it stopped being faithful within a day of the SDK being
written, which is the part worth internalising.

```bash
diff <(sed 's/[[:space:]]*$//' core/nodes/base_node.py) \
     <(sed 's/[[:space:]]*$//' ../Imagira-Node-SDK/src/imagira_node_sdk/base_node.py)
```

| Module | App | SDK 1.0.0 | SDK 1.1.0 |
|---|---|---|---|
| `base_node.py` | 315 lines | 280 — **executable divergence** | ✅ identical modulo comments |
| `expression.py` | 389 lines | 403 — faithful | ✅ still faithful |

The two divergences, both introduced in the app on **2026-07-25** by the fuzzing
campaign (`analysis/42_fuzz_findings.md`), **both now ported into SDK 1.1.0**:

1. **`as_bool_mask()` is new, and only in the app.** Masks legitimately arrive as
   `bool`, `uint8`, or a float soft matte, but only `bool` works as a selection —
   the fuzzer found 11 sites across 9 nodes that broke on the other two
   (`propagate` raised `ufunc 'invert' not supported`, `mask_similar` and
   `radiant` silently mis-indexed). The helper is now imported by **9 modules**:
   `core/nodes/removal/_shared.py`, `propagate`, `radiant`, `mask_similar`,
   `colour_filter`, `chroma_reduction`, `fill_masked`, `color_swatch`,
   `mean_std`. (`noise_estimation` fixes the same class inline with
   `mask.astype(bool)` and does not import the helper.)
2. **`mask_blend()` gained a shape guard in the app.** It now raises naming both
   arrays instead of letting numpy fail with
   `operands could not be broadcast together with shapes (96,96,1) (50,50,3)`.

Had the SDK been adopted against 1.0.0, **both fixes would have been reverted**:
the shim in Step 3 resolves `mask_blend` to whichever version the SDK ships, and
those 9 modules would have failed at import with `ImportError`. That does *not*
crash the app, which is the dangerous part: `_discover_node_classes` wraps
`importlib.import_module` in `except Exception` → `warnings.warn(...)` →
`continue`, so the affected node types quietly stop being registered. A warning
in a startup log is the only signal.

**Direction of the fix: forward, into the SDK** — never by reverting the app's
copy. Both changes are contract-level (every node author needs the mask-dtype
helper, and `mask_blend` was already in `__all__`), so both landed as a **minor**
bump:

- `as_bool_mask` — pure addition.
- The `mask_blend` guard — raises `ValueError`, the same exception type numpy
  already raised for those inputs, so the message improved and the contract
  didn't change.

Ported, exported, documented and tested in **SDK 1.1.0**; the parity test in
Step 1 is green against it. The rule this leaves behind: **a contract-level fix
lands in the SDK first, then arrives in the app through the dependency.**

## Step 1 — make drift detectable, not a thing you remember to check

A manual `diff` in a doc is not a mechanism; this drift happened inside one
working day, from an unrelated bug-fix pass. the app's `tests/test_sdk_parity.py`
runs it as a test, and skips the whole file when the SDK isn't importable — so it
costs nothing before adoption and holds the line after:

```bash
# against an installed SDK
pip install -e ../Imagira-Node-SDK && python -m pytest tests/test_sdk_parity.py -v

# or against a source checkout, without installing anything
IMAGIRA_SDK_SRC=../Imagira-Node-SDK/src python -m pytest tests/test_sdk_parity.py -v
```

Six checks. Against SDK 1.1.0: **5 pass, 2 skip, 0 fail** (the two skips are
gated on Steps 3-4 below). Against 1.0.0 the `base_node.py` row failed, printing
a diff of exactly what had to be ported — which is how Step 0 got found:

| Check | Today |
|---|---|
| `expression.py` identical modulo comments/docstrings/imports | ✅ passes — confirms the "still faithful" verdict mechanically |
| `base_node.py` identical | ✅ passes since SDK 1.1.0 — failed against 1.0.0 (`as_bool_mask` + the `mask_blend` guard) |
| every name in the SDK's `__all__` reachable from the app's modules | ✅ passes (author-facing `schema`/`testing` helpers excluded — Step 7 tracks those deliberately) |
| `core.nodes.base_node.BaseNode is imagira_node_sdk.BaseNode` | ⏭ skips until Step 3 — the identity check the migration exists to preserve |
| app `set_current_data` reaches an SDK node's `ExprParams` | ⏭ skips until Step 4 |
| two copies of `expression.py` do raise `ExpressionError` | ✅ passes — this one *documents* the pre-adoption failure mode and is deleted once the shims land |

That last pair is worth noting: the Step 4 table below was written from a manual
experiment, and the test now reproduces it on demand — binding through
`core.expression.set_current_data` and reading `{{ $json.v }}` through an
SDK-defined node raises `invalid expression {{ $json.v }}: attribute access only
allowed on dicts`, with nothing in the message pointing at a duplicated module.

## Step 2 — depend on it

```toml
# main app pyproject.toml
dependencies = ["imagira-node-sdk>=1.1,<2.0", ...]
```

`>=1.1` because of `as_bool_mask` (Step 0). Pinning `>=1.0` compiles and then
fails at import.

## Step 3 — make `core/nodes/base_node.py` a re-export shim

```python
# core/nodes/base_node.py
from imagira_node_sdk.base_node import (   # noqa: F401
    BaseNode, register, NODE_REGISTRY, ComputeBackend,
    to_rgb, mask_blend, as_bool_mask,
)
```

Every `from core.nodes.base_node import BaseNode, register` line in the built-in
nodes keeps working untouched, and there is now exactly one `BaseNode` class and
one `NODE_REGISTRY` object in the process regardless of which path a node
imported.

Do not hand-write this list from memory — derive it, or the next helper added to
one side goes missing on the other. `as_bool_mask` is exactly how that happens.

## Step 4 — do the same for `core/expression.py`. This is the subtle one.

`ExprParams` resolves templates against a module-level
`contextvars.ContextVar`, and the pipeline binds it via `set_current_data` right
before each `execute()`. **If the app keeps its own copy of that ContextVar while
nodes read the SDK's, the setter and the reader never meet and every node behaves
as if no data were bound, app-wide.** Confirmed by running both copies in one
process:

| Param value | With two copies loaded |
|---|---|
| `{{ $json }}` | `None`, silently |
| `{{ $json.fps }}` | `ExpressionError: invalid expression {{ $json.fps }}: attribute access only allowed on dicts` |
| `name_{{ $json.v }}.png` | same `ExpressionError` |
| `{{ $nodes['ck'].tolerance }}` | `ExpressionError: … subscript failed: 'ck'` |
| `plain_value` | unaffected |

It isn't wholly silent, then — but it's arguably worse: the error message blames
the user's expression, so what reaches you is "expressions are broken", with
nothing pointing at a duplicated module.

Hence `core/expression.py` becomes a shim too:

```python
# core/expression.py
from imagira_node_sdk.expression import (   # noqa: F401
    ExprParams, ExpressionError, evaluate, make_env, resolve_template,
    set_current_data, reset_current_data,
)
```

That covers the app's full public surface — `core/expression.py`'s exports are
exactly these seven names, re-verified 2026-07-25.

Importers that must keep working (re-verified against the app 2026-07-25 — all
9 rows still accurate, `api/expressions.py` now exists as an untracked new file):

| Importer | Names used |
|---|---|
| `core/pipeline.py` | `set_current_data`, `reset_current_data` |
| `core/nodes/base_node.py` | `ExprParams` |
| `core/nodes/utility/while_node.py` | `evaluate`, `ExpressionError`, `make_env` |
| `core/nodes/utility/if_else_node.py` | `evaluate`, `ExpressionError`, `make_env` |
| `api/expressions.py` | `ExpressionError`, `resolve_template` |
| `tests/test_expression_pow.py` | `ExpressionError`, `evaluate` |
| `tests/test_expression_templates.py` | `ExprParams`, `resolve_template`, `set_current_data`, `reset_current_data`, `ExpressionError` |
| `tests/test_control_flow.py` | `evaluate`, `ExpressionError`, `make_env` |
| `tests/test_expressions_api.py` | via the HTTP endpoint, not the module |

Anything the app has that isn't part of the contract — the control-flow *nodes*
built on `evaluate`, the expression-preview endpoint — stays in the app. Only
the module itself moves.

## Step 5 — the loader needs no changes

`core/nodes/__init__.py` already does `from core.nodes.base_node import
BaseNode, register, NODE_REGISTRY`, which now resolves to the SDK's objects
through the shim. `_discover_node_classes` and `_is_node_class` keep working for
in-tree and SDK-authored nodes alike, because both are now provably the same
class. The `IMAGIRA_ENABLE_CODE_NODES` gate is untouched.

One caveat worth knowing before you rely on the fault tolerance: discovery
catching and logging a failed import is what makes a broken *plugin* harmless,
but it is also what would make a broken *shim* quiet. A missing re-export
(Step 3) removes built-in nodes from the registry with a log line, not a
traceback. That's what the registry-parity check in Step 6 is for.

## Step 6 — prove it, don't assume it

An import test proves nothing here — one failure mode is silent and the other
misattributes itself to the user. Four checks:

1. **Registry parity.** `tests/test_node_smoke.py` already smoke-tests every
   registered type — `NODE_REGISTRY` must come out with exactly the same set of
   `type` keys as before the migration. Snapshot it first: 322 types registered
   as of 2026-07-25 (281 of them fuzzable, see `scripts/fuzz_workflows.py`).
2. **A pure-SDK node registers into the app's registry** and executes correctly
   through a real pipeline run, with zero `core.*` imports in its own module.
   This is the actual proof the contract didn't fork.
3. **The ContextVar is shared.** Bind data via the *app's*
   `core.expression.set_current_data`, then read a `{{ $json.x }}` param through
   an *SDK-defined* node's `self.params.get()` and assert it resolved:

   ```python
   from core.expression import set_current_data, reset_current_data
   from imagira_node_sdk import BaseNode

   class _Probe(BaseNode):
       type = "ctxvar_probe"
       param_schema = [{"key": "p", "label": "P", "type": "text", "default": ""}]
       def execute(self, image, mask=None, data=None, context=None):
           return {"image": image}

   node = _Probe("n1", params={"p": "{{ $json.v }}"})
   token = set_current_data({"v": "resolved"})
   try:
       assert node.params.get("p") == "resolved"   # fails if there are two copies
   finally:
       reset_current_data(token)
   ```

4. **Fault tolerance still holds.** A plugin that raises on import is caught,
   logged and skipped — never a startup crash. And `/api/plugins` still reports
   which package contributed which types and which crashed.

Checks 1 and 3 are implemented in `tests/test_sdk_parity.py` (skipped until the
SDK is installed). 2 and 4 need the dependency in place first.

## Step 7 — the SDK is a superset. Adopt the two modules the app doesn't have.

`schema.py` and `testing.py` have no app counterpart, and they are the part of
this migration that pays for itself rather than merely deduplicating code.

`validate_param_schema()` exists because **an unrecognised param type does not
error — `ParamInput.tsx`'s `switch` falls through `default:` to a plain text
input**, and nothing anywhere reports it. Run over the app's own registry today:

```bash
PYTHONPATH=../Imagira-Node-SDK/src:. python -c "
from core.nodes import NODE_REGISTRY
from imagira_node_sdk.schema import validate_param_schema
for t, cls in sorted(NODE_REGISTRY.items()):
    for issue in validate_param_schema(cls.param_schema):
        print(t, issue)"
```

**17 issues across 12 nodes**, all pre-existing:

| Issue | Nodes | Effect |
|---|---|---|
| `type: "int"` (9×) | `accumulate_node`, `for_each_start`, `while_node`, `pdf_source` ×3, `psd_source`, `image_sequence_sink` ×2 | renders as free text instead of a number field — no min/max, no steppers |
| `type: "image"` (2×) | `ssim_compare`, `tile_fill` | renders as free text; there is no image-picker branch at all |
| `type: "textarea"` (2×) | `try_catch_node`, `while_node` | a JSON sub-graph edited in a single-line input |
| `type: "color"` (1×) | `isolate_masked` | text box where every other node gets a colour picker |
| unrecognised key `description` | `curves` | ignored by the host; `help` is the rendered field |
| `default` below `min` | `resize` (`width`, `height`) | see below |

The `resize` rows are the interesting ones: `width`/`height` declare `min: 1`
while their own help text says *"Set to 0 to derive from Height or use the Scale
factor"*. `NumberInput` passes `min` to the input and clamps arrow-key stepping
to it, so the documented auto mode is not reachable with the steppers and the
browser marks the default invalid. The schema is wrong, not the code — `min: 0`.

`type: "int"` → `"number"` and `"color"` → `"colour"` are presentational
renames: the declared type never reaches stored param *values*, so saved
workflows are unaffected. `"image"` and `"textarea"` are different — they name
widgets the renderer doesn't have, so those four are a **feature** request
(image picker, multiline input), not a rename. Left as follow-up work; none of
this blocks adoption, and it is all visible today without the SDK — the SDK is
just what makes it *reportable*.

`testing.py` (`assert_node_contract`, `run_node`, `make_image`) overlaps
`tests/test_node_smoke.py` in intent but not in reach: the app's version guesses
one call signature and carries a 15-entry skip list, while the SDK's asserts the
contract per node without needing the pipeline. Worth pointing new built-in
nodes at it; not worth rewriting the smoke test.

## Recommendation: promote `accepts_context` to a real `BaseNode` attribute

The graph runner gates the `context` kwarg on `getattr(node, "accepts_context",
False)` (`core/pipeline.py`), and 10+ built-in nodes set it — `folder_loader`,
`video_source`, `video_sink`, `image_mask_exporter`, `data_transform`,
`for_each`, `while`, `if`, `if_else`, `switch`. But the flag is declared on
**neither** `BaseNode`: not in the app's copy, not in the SDK's. It exists only
as a duck-typed lookup.

That means a third-party author who writes `def execute(self, image, mask=None,
data=None, context=None)` — the signature `BaseNode.execute` itself documents —
receives `None` for context forever, with nothing to explain why. Every other
port-adjacent capability (`accepts_mask`, `accepts_data`, `accepts_samples`) is a
declared attribute with a default; this one isn't, purely by omission.

The fix is the additive shape the SDK's versioning policy already prescribes: add
`accepts_context = False` to `BaseNode` alongside the other `accepts_*` flags. No
existing node changes behaviour — the ones that set it keep working, the ones
that don't already behave as `False`. It's a minor bump, and it has to land in
**both** repos in the same change or the parity test goes red, since this is an
executable difference rather than a docstring one.

Not done unilaterally here: it's a contract change to the extracted file, and the
parity test would fail the moment the SDK had it and the app didn't.

## What stays out of the SDK

- `core/lint.py` — pipeline-level structural validation over assembled
  pipelines of instantiated nodes. Not a node author's concern.
- `core/pipeline.py` — topological execution and graph serialization. The host's
  job. Note this now includes `_mask_shape_error()`, the mask/image resolution
  guard: it needs the graph to name *which upstream node* diverged, so it cannot
  live in the contract.
- The If/Switch/While *nodes* — application logic built on top of the contract.
- Anything Flask/HTTP/UI-facing.
- `scripts/fuzz_workflows.py` — needs the whole registry and the pipeline. The
  SDK's `testing.py` is the node-author-scale equivalent.

## Should the shims be permanent?

Recommended: yes. They cost nothing, and keeping them means the 100+ built-in
node files never have to be touched. Deprecating them buys a tidier import graph
in exchange for a large mechanical diff and a migration window where both paths
must work anyway.

## Order of work

1. ~~Port `as_bool_mask` + the `mask_blend` guard into the SDK; release 1.1.~~
   **Done** — SDK 1.1.0, `CHANGELOG.md`, 12 new tests, suite 173 passing.
2. Land `tests/test_sdk_parity.py` (written; skips until the SDK is installed).
3. `pip install -e ../Imagira-Node-SDK`, add the dependency, convert both modules to shims.
4. Run the parity test, the full suite (1124 passing as of 2026-07-25), and a
   fuzz sweep — `python scripts/fuzz_workflows.py -n 400 --ignore-file scripts/fuzz_known.txt`
   exercises 219 node types through the shimmed contract in ~90 s.
5. Then the schema warnings from Step 7, at leisure.
