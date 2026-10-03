# Versioning and compatibility

Semver, strictly. Pin `imagira-node-sdk>=1.0,<2.0` and additive updates arrive
automatically while a breaking change never can.

The promise covers `imagira_node_sdk.__all__` and `imagira_node_sdk.testing`'s
`__all__`. Anything else — private names, module layout, internals of the
evaluator — may change in a patch release.

## Frozen at 1.0

Breaking any of these breaks every node in existence, in-tree or third-party:

- **`type` semantics.** A permanent identifier; saved workflow JSON references
  nodes by this string.
- **The four-stream port model** — image / mask / samples / data — and the
  `accepts_*` / `produces_*` / `has_*_port` flag names. The host app's frontend
  reads these for port colouring and lint rules; a change here is a coordinated
  break across the whole ecosystem.
- **`execute(self, image, mask=None, data=None, context=None)`** and its return
  dict (`image`, `mask`, `data`, optionally `samples` and `_iterations`).
- **`__init_subclass__`'s required_inputs/outputs-vs-flags validation.** This
  decides which node *definitions* are valid at import time, so loosening or
  tightening it changes what loads.
- **`to_rgb` and `mask_blend` behaviour**, including the details nodes rely on:
  `to_rgb` returns an already-correct array unchanged rather than copying, and
  `mask_blend` returns `result` untouched when `mask is None`. Since 1.1.0
  `mask_blend` also raises `ValueError` when the mask doesn't cover the result —
  the same exception type numpy already raised for those inputs, so the message
  is what changed, not the contract.
- **`as_bool_mask`'s thresholds** (since 1.1.0): `uint8` is `> 0`, a float matte
  is `> 0.5`, and a `bool` array is returned as-is rather than copied. Nodes
  branch on the result, so moving the float threshold would silently change
  which pixels they touch.
- **`ExprParams` read/write asymmetry.** `.get()` and `[...]` resolve
  templates; `.items()` / `.values()` / `dict(params)` read raw storage. That
  asymmetry is what makes `to_dict()` save `{{ }}` text instead of a resolved
  value, so a saved workflow round-trips.

## Free to grow in 1.x

- **New class attributes with conservative defaults.** This is the shape every
  additive change should take — it's how the app grew `optional_image_port`,
  `is_preview_output` and `is_disk_exporter` without breaking a single existing
  subclass.
- **New optional keyword arguments with defaults.**
- **New utility functions and new `param_schema` types.**
- **New optional modules** (`params`, `kinds`, 1.2) and optional schema fields (`legacy_options`).
- **New class attributes** such as `thread_safe_attrs` (1.2).
- **New checks in `validate_param_schema` / `assert_node_contract`.** These are
  opt-in test-time tools, not import-time gates, so a new check can only ever
  fail a test — never break a running node. That's precisely why
  `__init_subclass__` does *not* call the schema validator: turning schema
  smells into import errors would newly reject nodes that load fine today, which
  is a breaking change wearing a bugfix's clothes.

Never in a 1.x: renaming or removing an attribute, repurposing one, or changing
what an existing value means.

## Reserved for 2.0

Anything that requires existing third-party nodes to change code:

- Changing `execute()`'s parameter names or order.
- Renaming or removing a port flag.
- Making `validate_param_schema` mandatory at class definition time.
- Changing `type` mutability rules.
- Turning `self.params` into something that isn't an `ExprParams`.

## Why there must be exactly one copy of this package

Two subtle failure modes make "just vendor a copy" a bad idea:

**Two `BaseNode` classes → two registries.** If a host app keeps its own copy of
the contract instead of re-exporting the SDK's, then `issubclass(YourNode,
TheirBaseNode)` is `False`, the discovery loop doesn't recognise your node as a
node at all, and it never registers. Nothing errors.

**Two `expression` modules → templates stop resolving, and the error blames the
user.** `ExprParams` resolves against a module-level `contextvars.ContextVar`
that the host binds around each `execute()` call. If the host binds *its* copy's
ContextVar and your `ExprParams` reads *another* copy's, the two never meet and
every node behaves as if no data were bound: a whole-field `{{ $json }}` silently
yields `None`, while any path access raises `ExpressionError: invalid expression
{{ $json.fps }}: attribute access only allowed on dicts` — which reads as "the
user typed a bad expression", not "this app has two copies of one module". Nobody
debugging that message would look for a packaging problem.

Both are why Imagira depends on this package and re-exports from it rather than
keeping its own copy. See [host_integration.md](host_integration.md).

## Deprecation

If something ever has to go, it gets: a `DeprecationWarning` for at least one
minor release, a CHANGELOG entry under "contract changes", and removal only in
the next major.
