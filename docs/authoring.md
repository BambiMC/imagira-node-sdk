# Writing a node

A node is a Python class. Subclass `BaseNode`, set some attributes, implement
`execute()`. Nothing else is required.

```python
from imagira_node_sdk import BaseNode, register

@register
class MyNode(BaseNode):
    type = "myauthor_my_node"
    label = "My Node"

    def execute(self, image, mask=None, data=None, context=None):
        return {"image": image, "mask": mask}
```

## Identity and presentation

| Attribute | Meaning |
|---|---|
| `type` | **Permanent** unique identifier. See the warning below. |
| `label` | Display name in the palette and node header. |
| `category` | Grouping in the palette (`"Detection"`, `"Removal"`, `"Post-Process"`, `"Other"`, …). |
| `color` | Hex accent colour for the node card. |
| `description` | One line, shown in the palette and params panel. |
| `i18n` | Optional translation overrides. |

### `type` is permanent — treat it like a database primary key

Saved workflow JSON references nodes by this string. Change it and every saved
workflow that used your node stops resolving. Pick it once.

**Prefix it with your author or package name.** `NODE_REGISTRY` is a single flat
`dict[str, type]` shared by every node in the process, and the host app's loader
registers a type only `if node_type not in NODE_REGISTRY` — so if another
package already claimed `blur`, yours is dropped with no error and no log line
you'll notice. Nothing in the SDK or the app enforces namespacing; it is purely
a convention, which is exactly why you should follow it.

Keep it lowercase, no spaces, underscores between words.

### `i18n`

```python
i18n = {
    "de": {
        "label": "Mein Node",
        "description": "…",
        "params": {"amount": {"label": "Stärke", "help": "…"}},
    },
}
```

English class attributes are canonical; this dict only holds overrides, and
missing keys fall back silently. Option *values* are never translated.

**Define your own `i18n = {...}`.** It's a shared class-level dict — mutating
the inherited one at runtime leaks your translations onto every other node in
the process.

## Ports

Four independent streams flow between nodes: **image** (blue), **mask**
(orange), **samples** (red), **data** (green). Which ports your node shows is
declared with flags:

```python
has_input_port   = True    # blue image in       (default True)
has_output_port  = True    # blue image out      (default True)
produces_image   = True    # this node makes a new image, not a pass-through
accepts_mask     = False   # orange mask in
produces_mask    = False   # orange mask out
accepts_samples  = False   # red samples in
produces_samples = False   # red samples out
accepts_data     = False   # green data in
produces_data    = False   # green data out
```

See [data_streams.md](data_streams.md) for what actually travels over each.

### Required ports

`required_inputs` / `required_outputs` mark which ports are essential rather
than optional — the UI draws a filled port dot and the app's lint reports an
error if a required input is left disconnected.

```python
accepts_samples = True
required_inputs = ("samples",)   # this node is useless without them
```

The image port is auto-marked required by the host app whenever the node has it
and isn't a source or sink; set `optional_image_port = True` to opt out (e.g. a
routing node that also works as a pure mask/data pass-through).

**A required port must also be accepted or produced.** Declaring
`required_inputs = ("samples",)` without `accepts_samples = True` raises
`TypeError` at *class definition time* — the moment your module is imported,
before any pipeline runs. That's deliberate: a misconfigured node fails loudly
and legibly instead of halfway through someone's batch job.

### Topology flags

```python
is_source_node    = False   # generates its own images (no upstream image)
is_folder_loader  = False   # source that batch-loads from a folder
is_preview_output = False   # ends a preview-only pipeline
is_disk_exporter  = False   # writes files to disk
is_sink_node      = False   # terminal — no output port
```

These let the host app reason about graph shape without hardcoding type
strings.

## `execute()`

```python
def execute(self, image, mask=None, data=None, context=None):
    ...
    return {"image": out, "mask": mask}
```

**In:**

- `image` — `np.uint8` `(H, W, 3)` RGB. Run it through `to_rgb()` if you want
  to be safe against a grayscale/RGBA/float upstream.
- `mask` — `np.bool_` `(H, W)`, or `None`. May also arrive as `uint8` or a float
  matte from an upstream node — `as_bool_mask()` normalises it.
- `data` — `dict` or `None`, producer-defined keys.
- `context` — `dict` of shared per-run state.

### The last three arguments are opt-in — this trips people up

The host calls `execute(image, mask, **kwargs)` with **image and mask always
positional**, and adds the rest only when the matching class flag says the node
wants them:

| Argument | Passed only if | Default if not |
|---|---|---|
| `samples` | `accepts_samples = True` | never passed at all |
| `data` | `accepts_data = True` | never passed at all |
| `context` | `accepts_context = True` | never passed at all |

Two consequences worth internalising:

**`samples` is a fifth parameter that isn't in the default signature.** A node
that consumes the red wire declares it explicitly:

```python
accepts_samples = True
required_inputs = ("samples",)

def execute(self, image=None, mask=None, samples=None, data=None, context=None):
    ...
```

Set `accepts_samples = True` without adding the parameter and the host raises
`TypeError` on the node's first run. `assert_node_contract()` catches that.

**`context` needs `accepts_context = True`, which is not a `BaseNode`
attribute.** It's read off the class with `getattr(node, "accepts_context",
False)`, so it doesn't appear in the attribute list above and nothing validates
it. Declaring `context=None` in your signature is *not* enough — without the flag
your node silently receives `None` forever. If you need per-run context, set the
flag.

See [examples/sample_mask_filter_node.py](../examples/sample_mask_filter_node.py)
for a node that opts into `samples`.

**Out:** a dict. Recognised keys are `image`, `mask`, `data`, `samples`, and
`_iterations`; anything else is dropped silently by the pipeline. Return the
mask you were given if you didn't change it — downstream nodes need it.

The host pipeline calls this by keyword (`execute(image, mask=…, data=…,
context=…)`), so accept all four parameters (or `**kwargs`) even if you ignore
three of them.

**Return `uint8`.** Finishing in float means values silently wrap or clip
somewhere downstream. `np.clip(out, 0, 255).astype(np.uint8)`.

### GPU

```python
supports_gpu = True

def execute_gpu(self, image, mask=None, **kwargs):
    ...
```

Called instead of `execute()` when the run is dispatched to the GPU backend
*and* `supports_gpu = True`. The default implementation falls back to
`execute()`, so leaving it out is safe. Inputs and outputs are always CPU numpy
arrays — transfer is your node's job, and the SDK ships no transfer helpers.

## Params

```python
param_schema = [
    {"key": "amount", "label": "Amount", "type": "slider",
     "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
     "help": "How much of the effect to apply."},
]
```

Full spec in [param_schema.md](param_schema.md). Two rules matter here:

**Read params with `self.params.get(key, default)`.** `self.params` is an
`ExprParams`, a dict subclass whose `.get()` and `[...]` resolve `{{ }}`
templates on read. A user can type `{{ $json.fps }}` into any field and it
resolves against the incoming data payload, or `{{ $nodes["chroma_1"].tolerance
}}` to mirror another node's setting — with no work on your part, as long as you
read through `.get()`.

**Don't mutate `param_schema` or `_DEFAULT_PARAMS`.** They're class-level.
Defaults are cached from your schema once per subclass at definition time.

### Param descriptors (`imagira_node_sdk.params`)

Instead of dicts you can declare params as class attributes; the key defaults to
the attribute name and `to_schema()` yields the same dict shape:

```python
from imagira_node_sdk.params import Slider, Select

class MyNode(BaseNode):
    amount = Slider(0.5, 0.0, 1.0, help="How much of the effect to apply.")
    mode = Select("soft", ["soft", "hard"], help="Blend mode.")
```

Other additions: `legacy_options` (values a `select` still accepts from old
saved workflows without offering them); out-of-range `slider`/`number` values
are clamped on read; `thread_safe_attrs` lists attributes that may be set on
`self` during `execute()` (instances are shared across the image thread pool).

### Kinds (`imagira_node_sdk.kinds`)

Kind base classes (`Filter`, `MaskOp`, `Sampler`, `Analyzer`, `DataOp`, `Merge`,
`Source`, `Sink`, ...) fill in the port flags and `execute()` plumbing so you
implement only `apply()` (or the kind's hook). `accepts_data` is derived from
`param_schema` plus whether `execute` takes `data`, unless you set it.

**Import-time checks.** A `True` `accepts_samples`/`accepts_data`/`accepts_context`
flag requires a matching `execute()` parameter (or `**kwargs`), otherwise
`TypeError`. A `show_if` must reference a param declared *earlier*.

## Testing

`imagira_node_sdk.testing` exists so you can test a node with no Imagira
install at all:

```python
from imagira_node_sdk.testing import (
    assert_node_contract, run_node, make_image, make_mask, make_samples,
    isolated_registry,
)

def test_contract():
    assert_node_contract(MyNode)          # every problem reported at once

def test_behaviour():
    out = run_node(MyNode, image=make_image(4, 4, (10, 20, 30)),
                   params={"amount": 1.0})
    assert out["image"][0, 0].tolist() == [245, 235, 225]

def test_template_param():
    # exactly what happens when a user types {{ $json.x }} into a field
    out = run_node(MyNode, params={"amount": "{{ $json.strength }}"},
                   data={"strength": 0.5})
```

- `assert_node_contract(cls)` — checks the flags agree, `type` is set and
  namespaced, `execute()` is overridden with a workable signature, the schema is
  sane, `i18n` isn't the inherited dict. Pass `allow_schema_warnings=False` to
  also fail on schema smells.
- `run_node(cls, …)` — instantiates, binds the `{{ }}` data context the way the
  real pipeline does, calls `execute()`, and validates the returned dict's
  shape. That binding matters: without it your test sees raw template text and
  passes for the wrong reason. Pass `check_result=False` to skip validation.
- `isolated_registry()` — context manager that restores `NODE_REGISTRY` on
  exit, so `@register` inside a test doesn't leak into later tests.

Also worth a test of its own:

```python
from imagira_node_sdk import validate_param_schema

def test_schema_is_clean():
    assert validate_param_schema(MyNode) == []
```

## Then package it

See [packaging.md](packaging.md).
