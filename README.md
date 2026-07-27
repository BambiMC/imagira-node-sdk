# imagira-node-sdk

The plugin contract for building [Imagira](https://github.com/BambiMC/Imagira) pipeline
nodes — `BaseNode`, the port/param model, and the `imagira.nodes` entry-point
mechanism — as a small, independently-versioned package.

> **"Node" here means *graph node*** (one step in an image-processing
> pipeline), not Node.js. This is a Python package.

```bash
pip install imagira-node-sdk
```

numpy is the only runtime dependency. You do not need Imagira installed to
write or test a node.

## Why this exists

Before the split, subclassing `BaseNode` meant importing from Imagira's `core`
package — so every plugin dragged in Flask, opencv and scipy, and every
plugin's compatibility was tied to the app's internal release cadence. The
actual contract a node has to honour is small and stable: some class
attributes and one method. That's what this package is.

Imagira itself depends on this package and re-exports from it, so there is
exactly one `BaseNode` class and one `NODE_REGISTRY` per process — a node
written against the SDK and a node living in-tree are provably the same kind of
thing.

## A node in full

```python
import numpy as np
from imagira_node_sdk import BaseNode, register, to_rgb, mask_blend

@register
class InvertNode(BaseNode):
    type        = "myauthor_invert"     # permanent id — prefix it (see below)
    label       = "Invert"
    category    = "Post-Process"
    description = "Inverts the image, optionally only inside a mask."
    accepts_mask = True

    param_schema = [
        {"key": "amount", "label": "Amount", "type": "slider",
         "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    ]

    def execute(self, image, mask=None, data=None, context=None):
        amount = float(self.params.get("amount", 1.0))
        src = to_rgb(image)
        out = (255.0 - src) * amount + src * (1.0 - amount)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return {"image": mask_blend(src, out, mask), "mask": mask}
```

Test it without installing Imagira:

```python
from imagira_node_sdk.testing import assert_node_contract, run_node, make_image

def test_contract():
    assert_node_contract(InvertNode)

def test_inverts():
    out = run_node(InvertNode, image=make_image(2, 2, (10, 20, 30)))
    assert out["image"][0, 0].tolist() == [245, 235, 225]
```

Ship it by declaring an entry point — a running Imagira instance discovers it
on startup:

```toml
[project]
dependencies = ["imagira-node-sdk>=1.0,<2.0"]

[project.entry-points."imagira.nodes"]
my_package = "my_pkg.nodes"
```

Five complete, tested nodes live in
[examples/](https://github.com/BambiMC/Imagira-Node-SDK/tree/main/examples), each
covering a different shape of node:

| Example | Demonstrates |
|---|---|
| `invert_node.py` | the common case — image in, image out, params, mask blending, i18n |
| `constant_color_node.py` | a **source** (`is_source_node`, no image input) |
| `binary_segment_node.py` | a **mask producer** (`required_outputs = ("mask",)`) |
| `dominant_color_node.py` | a **samples producer** — the island payload's exact shape |
| `sample_mask_filter_node.py` | a **samples consumer** — the opt-in `samples` argument |

## Three things that bite people

1. **`type` is permanent.** Saved workflow JSON references nodes by this
   string. Renaming it breaks every workflow that used the node.
2. **Prefix your `type`.** `NODE_REGISTRY` is one flat, process-wide dict. Two
   packages claiming `blur` collide, and the loader resolves it as
   first-import-wins — silently.
3. **Read params via `self.params.get("key", default)`.** `self.params` is an
   `ExprParams`, not a plain dict: reading through `.get()` / `[...]` is what
   resolves a `{{ $json.fps }}` template a user typed into the field. Reaching
   into the underlying storage bypasses that and hands you the raw text.

## Docs

| | |
|---|---|
| [docs/authoring.md](https://github.com/BambiMC/Imagira-Node-SDK/blob/main/docs/authoring.md) | Writing a node: attributes, `execute()`, params, testing |
| [docs/param_schema.md](https://github.com/BambiMC/Imagira-Node-SDK/blob/main/docs/param_schema.md) | The `param_schema` spec — types, keys, and the silent text fallback |
| [docs/data_streams.md](https://github.com/BambiMC/Imagira-Node-SDK/blob/main/docs/data_streams.md) | image / mask / samples / data — types and shapes |
| [docs/packaging.md](https://github.com/BambiMC/Imagira-Node-SDK/blob/main/docs/packaging.md) | Entry points, layout, publishing, `type` namespacing |
| [docs/versioning.md](https://github.com/BambiMC/Imagira-Node-SDK/blob/main/docs/versioning.md) | What 1.0 freezes, what may still be added, how to pin |
| [docs/host_integration.md](https://github.com/BambiMC/Imagira-Node-SDK/blob/main/docs/host_integration.md) | For host apps: how to depend on the SDK without forking the contract |

## Public API

Everything in `imagira_node_sdk.__all__` is covered by the semver promise:

`BaseNode` · `register` · `NODE_REGISTRY` · `ComputeBackend` · `to_rgb` ·
`mask_blend` · `as_bool_mask` · `ExprParams` · `ExpressionError` · `resolve_template` ·
`set_current_data` · `reset_current_data` · `evaluate` · `make_env` ·
`validate_param_schema` · `SchemaIssue` · `CANONICAL_PARAM_TYPES` ·
`__version__`

Plus `imagira_node_sdk.testing` (`assert_node_contract`, `run_node`,
`make_image`, `make_mask`, `make_samples`, `isolated_registry`), which is
deliberately not imported by the package root so it stays out of a production
instance's runtime path.

Anything not listed is an implementation detail and may change in a patch
release.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src/imagira_node_sdk
```

MIT licensed.
