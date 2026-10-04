# Changelog

All notable changes to `imagira-node-sdk`.

This project follows [semantic versioning](https://semver.org/) as described in
[docs/versioning.md](docs/versioning.md). **Contract changes** — anything
affecting what a node must do to keep working — are listed separately from
internal changes, because those are the only ones that can cost a node author
anything.

## [1.2.0] — unreleased

Catch-up port of the app's vendored modules (`analysis/54_quality.md` §5.13).
`base_node.py` and `expression.py` are re-synced and two modules are new:
`imagira_node_sdk.params` and `imagira_node_sdk.kinds`. `tests/test_sdk_parity.py`
in the app diffs all four. Nothing was added to `__all__`; use the submodules.

### Contract

- **New modules.** `params` (descriptors `Param`, `Slider`, `Number`, `Bool`,
  `Select`, `Colour`, `Text`, `Raw`, plus `from_schema`, `params_from_class`,
  `ParamView`) and `kinds` (`Filter`, `Geometry`, `MaskMaker`, `MaskOp`,
  `Sampler`, `Analyzer`, `DataOp`, `Merge`, `SampleConsumer`, `Source`, `Sink`,
  `masked_pixels`). Both are additive; they declare ordinary `BaseNode`s/schemas.
- **New `BaseNode` attributes:** derived `accepts_data`, `thread_safe_attrs`,
  optional schema field `legacy_options`; `ExprParams(..., coerce=...)` with
  clamp-on-read for `slider`/`number`.
- **Strict-mode state guard**: `_state_on_self` / `_EXECUTING`; a no-op without
  the app's `core.node_contract`.

### Fixed (expression evaluator, mirrored from the app)

- `evaluate()` raises `ExpressionError` for runtime failures (`1/0`, `int('abc')`,
  `len(5)`, runaway recursion) instead of leaking `ZeroDivisionError`/`ValueError`/
  `TypeError`; callers only catch `ExpressionError`.
- `and` / `or` short-circuit, so `$x != 0 and 10 / $x > 2` is safe.
- Sequence repetition is capped (`[0]*10**9`, `'a'*10**10` raise `ExpressionError`);
  only `**` was bounded before.
- `register()` logs a warning when a different class replaces an already-registered
  node `type` (last registration still wins; behaviour is unchanged).
- The `Analyzer` kind gates on the same boolean selection that indexes the pixels, so a
  soft mask whose values are all <= 0.5 falls back to the whole frame instead of an
  empty measurement.

### Compatibility (why this is a minor release)

- A `True` `accepts_samples` / `accepts_data` / `accepts_context` flag whose
  `execute()` lacks that parameter (and `**kwargs`), and a `show_if` that gates
  on a param listed at or after the gated one, are reported at class creation.
  They raise `TypeError` only under strict mode (`IMAGIRA_STRICT_CONTRACT=1`,
  which the host sets for its tests and fuzzers); otherwise the host logs them
  once and the SDK on its own emits a `UserWarning`. A 1.1 plugin still loads.
  The engine already passes a kwarg only when the signature takes it, so the
  flag mismatch is harmless at run time.
- An unknown param `type` (the `color` typo for `colour`) raises at class
  creation. The host has rejected such classes since before 1.2; the SDK's
  copy was simply older, so no plugin that loads in the host is affected.
- The `config.NODE_CATEGORIES` check only fires for classes in `core.nodes.*`,
  so it never applies to plugins.

## [1.1.0] — unreleased

Two additions ported **from** the main app, in the opposite direction to 1.0.0's
extraction. Both came out of Imagira's first fuzzing campaign (2026-07-25), which
found them by composing random graphs out of the whole node registry; both are
contract-level rather than host-specific, so keeping them app-side would have
forked the contract on day one. See `docs/host_integration.md` Step 0.

### Contract

- **`as_bool_mask(mask)`** — new export. Returns a strict boolean view of a mask:
  `bool` unchanged, `uint8` as `> 0`, a float soft matte as `> 0.5`, `None` as
  `None`. Only a bool mask can be used as a *selection* — `image[mask]` with a
  uint8 mask is integer indexing, and `~mask` on a float matte raises
  `ufunc 'invert' not supported`. The fuzzer found this in **nine** shipped
  Imagira nodes at once (`propagate`, `radiant`, `mask_similar`, `colour_filter`,
  `chroma_reduction`, `fill_masked`, `color_swatch`, `mean_std`, and the shared
  feather helper). Additive: no existing call changes behaviour.
- **`mask_blend` now raises on a mask that doesn't cover the result**, naming
  both shapes and what to return instead. Previously numpy failed with
  `operands could not be broadcast together with shapes (96,96,1) (50,50,3)`,
  which names neither the node nor which side is wrong. **Not a breaking
  change**: it raises `ValueError`, the same exception type numpy already raised
  for exactly those inputs — only the message improves. Matching shapes and
  `mask is None` are untouched.

### Docs

- `docs/data_streams.md` — new "Reading a mask: convert before you index"
  section. The mask stream doc said what to *return* but never what you might
  *receive*, which is the gap both bugs above fell through.
- `docs/versioning.md` — `as_bool_mask`'s thresholds and `mask_blend`'s guard
  added to the frozen-behaviour list.
- `docs/host_integration.md` — rewritten Step 0/1: the extraction had already
  drifted, and drift is now enforced by a parity test in the app repo
  (`tests/test_sdk_parity.py`) rather than a `diff` command in a doc.

### Contract documentation — `execute()`'s opt-in arguments

Writing the new examples surfaced a gap: the documented signature
`execute(self, image, mask=None, data=None, context=None)` is not how the host
actually calls a node, and nothing in the SDK said so.

- **`samples` is a fifth, opt-in parameter.** The host calls
  `execute(image, mask, **kwargs)` — image and mask positional — and adds
  `samples`, `data` and `context` only when the node's matching `accepts_*` flag
  is True. A samples-consuming node must declare both the flag and the
  parameter; setting the flag alone is a `TypeError` on first run.
- **`context` requires `accepts_context = True`**, which is read off the class
  with `getattr(node, "accepts_context", False)` and is **not** a `BaseNode`
  attribute in either repo. Declaring `context=None` in a signature without the
  flag means the argument is silently always `None`. Documented in
  `docs/authoring.md`; see the note in `docs/host_integration.md` about
  promoting it to a real attribute.

### Test kit

- **`run_node()` now mirrors the host's calling convention** rather than passing
  all four arguments by keyword. It gained a `samples=` parameter, and gates
  `samples`/`data`/`context` on the same `accepts_*` flags the host reads. The
  old behaviour let a test pass for a node the app would raise `TypeError` on.
- **`assert_node_contract()` no longer demands `mask`/`data`/`context` in the
  signature.** That was a false positive on the canonical short signature real
  nodes use. It now checks flag-to-signature agreement in the direction that
  actually breaks: `accepts_X = True` with no matching parameter.

### Examples

Four more, each covering a shape the original invert example didn't, all adapted
from real Imagira built-ins and depending on nothing but the SDK and numpy:
`constant_color_node.py` (a source), `binary_segment_node.py` (a mask producer),
`dominant_color_node.py` (a samples producer, with the island payload's exact
shape), `sample_mask_filter_node.py` (a samples consumer — the one that shows
the opt-in `samples` argument).

### Tests

- 12 new cases in `tests/test_base_node.py` (`TestAsBoolMask`,
  `TestMaskBlendShapeGuard`).
- `tests/test_examples.py` — 44 cases covering the four new examples, plus
  contract and schema checks parametrised across all of them.
- Suite: 222 passing on 3.10 and 3.11.

## [1.0.0] — unreleased

First release. Freezes the plugin contract as it exists in the main Imagira app
today, rather than inventing anything new.

### Contract

- `BaseNode` — extracted verbatim from Imagira's `core/nodes/base_node.py`:
  identity/presentation attributes (`type`, `label`, `category`, `color`,
  `description`, `i18n`), the port flags (`has_input_port`, `has_output_port`,
  `produces_image`, `accepts_mask`/`produces_mask`,
  `accepts_samples`/`produces_samples`, `accepts_data`/`produces_data`,
  `optional_image_port`, `required_inputs`, `required_outputs`), the topology
  flags (`is_source_node`, `is_folder_loader`, `is_preview_output`,
  `is_disk_exporter`, `is_sink_node`), `supports_gpu`, `param_schema`,
  `execute()`, `execute_gpu()`, `get_param_value()`, `set_param_value()`,
  `to_dict()`, `from_dict()`, and `_default_params()`.
- `__init_subclass__` validation of `required_inputs`/`required_outputs` against
  the port flags, raising `TypeError` at class definition time.
- `register`, `NODE_REGISTRY`, `ComputeBackend`.
- `to_rgb`, `mask_blend`.
- `ExprParams` and the `{{ }}` template layer (`resolve_template`,
  `set_current_data`, `reset_current_data`), vendored verbatim from
  `core/expression.py` — including the `**` magnitude cap that prevents a
  CPU/memory DoS through the evaluator. Vendored rather than left in the app so
  a node tested standalone behaves exactly as it does inside Imagira.
- `evaluate` / `make_env` — the condition evaluator, exported for host apps and
  control-flow node authors.

### New in the SDK (not present in the app)

- **`imagira_node_sdk.schema`** — the `param_schema` specification written down
  for the first time, plus an opt-in `validate_param_schema()`. It is *not*
  wired into `__init_subclass__`: making schema smells raise at import would
  newly reject nodes that load fine today. Notably it catches the
  silent-text-fallback class of bug, where an unrecognised `type` (e.g. `color`
  for `colour`) renders as a plain text input with no error anywhere.
- **`imagira_node_sdk.testing`** — `assert_node_contract`, `run_node`,
  `make_image`, `make_mask`, `make_samples`, `isolated_registry`. Lets a
  third-party node be tested with no Imagira install, which is the point of the
  split.
- `py.typed`.

### Notes

- numpy is the only runtime dependency, with a floor of `>=1.24` — lower than
  the main app's `>=2.0`, so installing a node package can't force an unrelated
  environment onto numpy 2. Nothing here uses a 2.x-only API.
- Published as 1.0.0 rather than 0.x deliberately: the value proposition is a
  surface authors can pin with `>=1.0,<2.0`, which 0.x can't express.
