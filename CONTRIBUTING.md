# Contributing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check . && mypy src/imagira_node_sdk
```

## The one rule

**Changes are additive.** Every third-party node package in existence pins
`imagira-node-sdk>=1.0,<2.0`, which means anything that makes an existing node
stop working is not a bugfix here — it's a 2.0.

New attribute with a conservative default: yes. New optional keyword argument
with a default: yes. New utility, new schema type, new opt-in check: yes.
Renaming an attribute, removing one, repurposing one, changing what a value
means, or tightening what loads at import time: no — open an issue instead, so it
can be batched into a major.

[docs/versioning.md](docs/versioning.md) lists exactly what's frozen. Read it
before proposing a change to `base_node.py` or `expression.py`.

## Two things that look like cleanups but aren't

**`__init_subclass__` does not call `validate_param_schema`.** That looks like an
obvious wiring-up, but it would newly reject nodes that import fine today — a
breaking change dressed as a bugfix. The validator is a test-time tool
deliberately.

**`to_dict()` uses `dict(dict.items(self.params))`, not `dict(self.params)`.**
The second form goes through `ExprParams.__getitem__` and would serialize
whatever a `{{ }}` template last resolved to into the saved workflow instead of
the template text. There's a regression test; don't "simplify" past it.

## Extraction fidelity

`base_node.py` and `expression.py` are verbatim copies of the main app's
`core/nodes/base_node.py` and `core/expression.py` — only comments, docstrings
and the one import line differ. Keep it that way: any behavioural change here
has to land in both repos, or the app's re-export shim silently means something
different from what a node author tested against.

```bash
diff <(sed 's/[[:space:]]*$//' ../Imagira/core/nodes/base_node.py) \
     <(sed 's/[[:space:]]*$//' src/imagira_node_sdk/base_node.py)
```

## Tests

New behaviour needs a test. Prefer one that fails *loudly* for the right reason
— several of the bugs this package guards against are silent (a template that
stops resolving, a param that renders as the wrong widget, a result key that gets
dropped), so a test that merely checks something didn't crash is not much use.

## Changelog

Add an entry under the unreleased heading, in the **Contract** section if a node
author could notice it.
