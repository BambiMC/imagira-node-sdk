# Packaging and publishing a node package

A running Imagira instance discovers third-party nodes through the
`imagira.nodes` entry-point group. Declare it, `pip install` your package into
the same environment, restart — the nodes appear in the palette.

## Layout

```
my-imagira-nodes/
├── pyproject.toml
├── README.md
├── src/
│   └── my_pkg/
│       ├── __init__.py
│       └── nodes.py          # imports from imagira_node_sdk
└── tests/
    └── test_nodes.py
```

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-imagira-nodes"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["imagira-node-sdk>=1.0,<2.0"]

[project.entry-points."imagira.nodes"]
my_package = "my_pkg.nodes"

[tool.setuptools.packages.find]
where = ["src"]
```

The pin matters: `>=1.0,<2.0` gets you additive SDK updates automatically and
never an unannounced breaking change. See [versioning.md](versioning.md).

## Entry-point targets

Two forms:

```toml
[project.entry-points."imagira.nodes"]
my_node    = "my_pkg.nodes:MyCoolNode"   # one class
my_package = "my_pkg.nodes"              # a module — every BaseNode subclass in it
```

For a module target the loader walks the module's classes and registers each
concrete `BaseNode` subclass it finds, so you don't have to list them
individually. The name on the left is a label used in diagnostics; it doesn't
have to match anything.

`@register` on the class is the in-module equivalent and is what in-tree nodes
use. Both end up in the same `NODE_REGISTRY`.

## Namespacing your `type` — do this

`NODE_REGISTRY` is a single flat `dict[str, type]`, process-wide, in whichever
app instance loaded the plugins. The loader registers a type only
`if node_type not in NODE_REGISTRY`, so **two packages claiming the same type
string collide and the first one imported wins, silently.** Nothing in the SDK
or the app prevents it.

```python
type = "myauthor_edge_glow"     # good
type = "edge_glow"              # a collision waiting to happen
```

## What happens on the host at startup

- The loader iterates the `imagira.nodes` entry points and imports each target.
- **A broken plugin cannot take down the app.** An import error or a bad class
  is caught, logged as a warning, and skipped. This is a deliberate hard
  requirement: an instance with a dozen third-party packages installed must
  still boot when one of them is broken.
- Diagnostics are surfaced by the app (which package contributed which types,
  and which crashed), so a user can see that your plugin failed rather than
  wondering where it went.

The corollary: **your node failing to appear is usually a silent skip, not a
crash.** The two most common causes are an exception at import time and a `type`
that collides with an already-loaded node. Check the app's plugin diagnostics
first.

## Version compatibility

There is currently **no compatibility gate in the loader** — nothing checks a
declared minimum SDK or Imagira version before importing your plugin. Your
dependency pin on `imagira-node-sdk` is what actually protects you, since pip
resolves it at install time. Declaring a minimum Imagira version is a
convention only; it isn't enforced anywhere yet.

## Testing before you publish

Your package's tests need nothing but the SDK — no Imagira install:

```python
from imagira_node_sdk import validate_param_schema
from imagira_node_sdk.testing import assert_node_contract, run_node, make_image
from my_pkg.nodes import MyNode

def test_contract():
    assert_node_contract(MyNode)

def test_schema_is_clean():
    assert validate_param_schema(MyNode) == []

def test_it_does_the_thing():
    out = run_node(MyNode, image=make_image(8, 8))
    assert out["image"].shape == (8, 8, 3)
```

Worth also checking that your entry point resolves in a real install, since a
typo'd module path is invisible until startup:

```bash
pip install -e .
python -c "
from importlib.metadata import entry_points
for ep in entry_points(group='imagira.nodes'):
    print(ep.name, '->', ep.load())
"
```

## Publishing

Standard Python packaging — `python -m build`, then `twine upload`, or PyPI
Trusted Publishing from CI. Nothing Imagira-specific.
