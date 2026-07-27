# `param_schema`

`param_schema` is how a node declares its settings. The host app turns it into
the params panel; the SDK turns it into `_DEFAULT_PARAMS`, which seeds every
instance's `self.params`.

```python
param_schema = [
    {"key": "amount", "label": "Amount", "type": "slider",
     "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
     "help": "How much of the effect to apply."},
    {"key": "mode", "label": "Mode", "type": "select",
     "default": "soft", "options": ["soft", "hard"]},
    {"key": "hardness", "label": "Hardness", "type": "number", "default": 3,
     "show_if": {"key": "mode", "values": ["hard"]}},
]
```

## The one thing to know first

**An unrecognised `type` is not an error — it silently renders as a plain text
input.** Write `"color"` instead of `"colour"` and your user gets a text box
where a colour picker belongs. Nothing logs, nothing warns, tests still pass.

So validate your schema in your own test suite:

```python
from imagira_node_sdk import validate_param_schema

def test_schema_is_clean():
    assert validate_param_schema(MyNode) == []
```

`validate_param_schema` returns a list of `SchemaIssue(level, where, message)`
— `level` is `"error"` (malformed; the app or `__init_subclass__` will
misbehave) or `"warning"` (loads fine, doesn't do what you meant). It never
raises, so you see every problem at once instead of one per test run.

It is **not** called automatically. Making schema smells raise at import would
newly reject nodes that load fine today — a breaking change disguised as a
bugfix. See [versioning.md](versioning.md).

## Types

Exactly these, from `CANONICAL_PARAM_TYPES`:

| `type` | Renders as | Notes |
|---|---|---|
| `slider` | Range slider | Needs `min` and `max` to have anything to draw |
| `number` | Numeric text field | Use for ints and floats alike |
| `select` | Dropdown | Requires `options` |
| `bool` | Checkbox | `default` should be a real `True`/`False` |
| `colour` | Colour picker + hex field | **British spelling.** `default` must be `#rrggbb` |
| `text` | Single-line text field | Also the fallback for anything unrecognised |
| `breakpoints` | Editable (value %, tolerance) table | For curve-shaped settings |

Common near-misses, all of which degrade to a text input:

| You might write | Use instead |
|---|---|
| `color` | `colour` |
| `int`, `float` | `number` |
| `string`, `str`, `textarea` | `text` |
| `boolean`, `checkbox` | `bool` |
| `dropdown` | `select` |
| `range` | `slider` |

## Keys

| Key | Applies to | Meaning |
|---|---|---|
| `key` | all | **Required.** The name your `execute()` looks up. Missing it raises `KeyError` when the class is defined. |
| `type` | all | **Required.** One of the table above. |
| `label` | all | Shown next to the input. Omit it and the row is unnamed. |
| `default` | all | Seeds `_DEFAULT_PARAMS`. **Absent means `None`** — `self.params.get(key)` returns `None` until the user touches the field. Always set one. |
| `min`, `max`, `step` | `slider`, `number` | Numeric bounds. `step` must be positive. |
| `options` | `select` | List of **plain strings** — each is used as both the value and the visible text. `{"value":…, "label":…}` objects do not work. |
| `help` | all | Tooltip / help text. |
| `placeholder` | `text` | Placeholder text. |
| `show_if` | all | `{"key": other_param, "values": [...]}` — hide unless `other_param`'s value is in `values`. `key` must name another param in the same schema. |
| `data_key` | all | Marks the param as fed from the incoming data port. |
| `browse` | `text` | `"folder"`, `"video"`, or `"output_folder"` — offers a server-side native file dialog. |
| `browse_default_name` | `text` with `browse="output_folder"` | Seed filename when the current value has none. |

Any other key is ignored silently by the host app, so `validate_param_schema`
flags unrecognised keys as a typo catcher (`"helptext"` instead of `"help"` is
a real and invisible bug otherwise).

## What the SDK does with it

At class definition time, `__init_subclass__` builds
`_DEFAULT_PARAMS = {p["key"]: p.get("default") for p in param_schema}` and
caches it on the subclass — once per class, not per instance, since a batch run
can instantiate a lot of nodes.

Each instance then gets `self.params = ExprParams(self._DEFAULT_PARAMS)`,
updated with whatever the saved workflow supplied. So:

- A param in the schema but never configured → its `default`.
- A param with no `default` → `None`.
- Reading via `self.params.get(key)` resolves `{{ }}` templates; the underlying
  storage keeps the raw text so saving a workflow round-trips the template, not
  its last resolved value.

## Full example

```python
param_schema = [
    {"key": "enabled", "label": "Enabled", "type": "bool", "default": True},

    {"key": "amount", "label": "Amount", "type": "slider",
     "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
     "help": "0 = no effect, 1 = full effect.",
     "show_if": {"key": "enabled", "values": [True]}},

    {"key": "mode", "label": "Mode", "type": "select",
     "default": "soft", "options": ["soft", "hard", "adaptive"]},

    {"key": "tint", "label": "Tint", "type": "colour", "default": "#00ff00"},

    {"key": "out_dir", "label": "Output folder", "type": "text", "default": "",
     "browse": "folder", "placeholder": "/path/to/output"},
]
```
