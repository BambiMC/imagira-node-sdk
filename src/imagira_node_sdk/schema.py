"""The ``param_schema`` specification, and an opt-in validator for it.

``param_schema`` is half the plugin contract — it's what the host app turns
into the params panel — but it was only ever specified by example. This module
writes it down.

The canonical type vocabulary here is taken from the host app's own param
renderer (its ``ParamType`` union), which is the thing that actually decides
how a param is drawn. That matters because **an unrecognised type does not
error — it silently falls back to a plain text input.** A node with
``"type": "color"`` (rather than the canonical ``"colour"``) gets a text box
where the user expected a colour picker, and nothing anywhere reports it.
Catching that is this module's main reason to exist.

Nothing here runs automatically. ``BaseNode.__init_subclass__`` deliberately
does *not* call it: turning schema smells into import-time errors would newly
reject nodes that load fine today, which is a breaking change dressed up as a
bugfix. Call it from your own test suite instead::

    from imagira_node_sdk.schema import validate_param_schema

    def test_schema_is_clean():
        assert validate_param_schema(MyNode) == []
"""

from __future__ import annotations

from dataclasses import dataclass

# ── The canonical vocabulary ────────────────────────────────────────────────
#
# Exactly the types the host app's param renderer has a branch for. Frozen for
# the 1.x line: additions are a minor bump, and nothing is ever removed.
CANONICAL_PARAM_TYPES: frozenset[str] = frozenset({
    "slider",       # float within a range, drawn as a slider (needs min/max)
    "number",       # int or float in a text field
    "select",       # dropdown over `options`
    "bool",         # checkbox
    "colour",       # colour picker — note the British spelling
    "text",         # single-line free text
    "breakpoints",  # editable (value %, tolerance) table
})

# Spellings that appear in real node code but that the renderer has no branch
# for — each one silently degrades to a text input. Mapped to what was meant.
KNOWN_MISSPELLINGS: dict[str, str] = {
    "color":    "colour",
    "int":      "number",
    "float":    "number",
    "string":   "text",
    "str":      "text",
    "boolean":  "bool",
    "checkbox": "bool",
    "dropdown": "select",
    "range":    "slider",
    "textarea": "text",
}

# Keys the host app reads off a schema entry. Anything else is a typo as far as
# the UI is concerned — it will be ignored without complaint.
KNOWN_ENTRY_KEYS: frozenset[str] = frozenset({
    "key",                  # param name — how your execute() looks it up
    "type",                 # one of CANONICAL_PARAM_TYPES
    "label",                # shown next to the input
    "default",              # seeds _DEFAULT_PARAMS; absent means None
    "min", "max", "step",   # numeric bounds (slider/number)
    "options",              # list[str] for select
    "help",                 # tooltip / help text
    "placeholder",          # text-input placeholder
    "data_key",             # marks the param as fed from the data port
    "show_if",              # {"key": other_key, "values": [...]} conditional
    "browse",               # "folder" | "video" | "output_folder"
    "browse_default_name",  # seed filename for browse="output_folder"
})

BROWSE_KINDS: frozenset[str] = frozenset({"folder", "video", "output_folder"})

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class SchemaIssue:
    """One problem found in a ``param_schema``.

    ``level`` is ``"error"`` (the schema is malformed — the host app or
    ``__init_subclass__`` will misbehave) or ``"warning"`` (it will load, but
    not do what the author meant — the silent-text-fallback class of bug).
    """

    level: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


def validate_param_schema(node_or_schema) -> list[SchemaIssue]:
    """Check a ``param_schema`` (or a node class carrying one).

    Returns a list of :class:`SchemaIssue`, empty if the schema is clean.
    Never raises for schema problems — errors are *returned* so a caller can
    print all of them at once instead of fixing one per test run.
    """
    schema = getattr(node_or_schema, "param_schema", node_or_schema)
    label = getattr(node_or_schema, "__name__", "param_schema")
    issues: list[SchemaIssue] = []

    if not isinstance(schema, (list, tuple)):
        return [SchemaIssue(ERROR, label,
                            f"param_schema must be a list, got {type(schema).__name__}")]

    seen_keys: set[str] = set()
    all_keys = {e["key"] for e in schema
                if isinstance(e, dict) and isinstance(e.get("key"), str)}

    for i, entry in enumerate(schema):
        where = f"{label}[{i}]"
        if not isinstance(entry, dict):
            issues.append(SchemaIssue(ERROR, where,
                                      f"entry must be a dict, got {type(entry).__name__}"))
            continue

        # ── key ────────────────────────────────────────────────────────────
        key = entry.get("key")
        if not isinstance(key, str) or not key:
            # This one really is fatal: __init_subclass__ does p["key"] while
            # building _DEFAULT_PARAMS, so the class raises KeyError on import.
            issues.append(SchemaIssue(ERROR, where,
                                      "missing or empty 'key' — BaseNode.__init_subclass__ "
                                      "will raise KeyError when the class is defined"))
        else:
            where = f"{label}[{i}] {key!r}"
            if key in seen_keys:
                issues.append(SchemaIssue(ERROR, where,
                                          "duplicate 'key' — the later entry silently wins "
                                          "in _DEFAULT_PARAMS"))
            seen_keys.add(key)

        # ── type ───────────────────────────────────────────────────────────
        ptype = entry.get("type")
        if ptype is None:
            issues.append(SchemaIssue(ERROR, where, "missing 'type'"))
        elif not isinstance(ptype, str):
            issues.append(SchemaIssue(ERROR, where,
                                      f"'type' must be a string, got {type(ptype).__name__}"))
        elif ptype not in CANONICAL_PARAM_TYPES:
            suggestion = KNOWN_MISSPELLINGS.get(ptype)
            hint = f" — did you mean {suggestion!r}?" if suggestion else ""
            issues.append(SchemaIssue(WARNING, where,
                                      f"unknown type {ptype!r}{hint} Unknown types are not "
                                      f"rejected; they silently render as a plain text input. "
                                      f"Canonical: {sorted(CANONICAL_PARAM_TYPES)}"))

        # ── label / default ────────────────────────────────────────────────
        if not entry.get("label"):
            issues.append(SchemaIssue(WARNING, where,
                                      "no 'label' — the params panel will have an unnamed row"))
        if "default" not in entry:
            issues.append(SchemaIssue(WARNING, where,
                                      "no 'default' — _DEFAULT_PARAMS gets None, so "
                                      "self.params.get(key) returns None until the user "
                                      "touches the param"))

        # ── numeric bounds ─────────────────────────────────────────────────
        for bound in ("min", "max", "step"):
            if bound in entry and not isinstance(entry[bound], (int, float)):
                issues.append(SchemaIssue(ERROR, where,
                                          f"'{bound}' must be a number, got "
                                          f"{type(entry[bound]).__name__}"))
        lo, hi = entry.get("min"), entry.get("max")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
            issues.append(SchemaIssue(ERROR, where, f"min ({lo}) is greater than max ({hi})"))
        step = entry.get("step")
        if isinstance(step, (int, float)) and step <= 0:
            issues.append(SchemaIssue(ERROR, where, f"'step' must be positive, got {step}"))
        if ptype == "slider" and not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
            issues.append(SchemaIssue(WARNING, where,
                                      "a 'slider' without both 'min' and 'max' has no range "
                                      "to draw"))
        default = entry.get("default")
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            if isinstance(lo, (int, float)) and default < lo:
                issues.append(SchemaIssue(WARNING, where,
                                          f"default ({default}) is below min ({lo})"))
            if isinstance(hi, (int, float)) and default > hi:
                issues.append(SchemaIssue(WARNING, where,
                                          f"default ({default}) is above max ({hi})"))

        # ── select / options ───────────────────────────────────────────────
        options = entry.get("options")
        if ptype == "select":
            if not options:
                issues.append(SchemaIssue(ERROR, where, "'select' requires a non-empty 'options'"))
            elif not isinstance(options, (list, tuple)):
                issues.append(SchemaIssue(ERROR, where,
                                          f"'options' must be a list, got {type(options).__name__}"))
            else:
                if not all(isinstance(o, str) for o in options):
                    issues.append(SchemaIssue(ERROR, where,
                                              "'options' must be a list of plain strings, not "
                                              "{value, label} objects — the renderer uses each "
                                              "entry as both the value and the visible text"))
                elif default is not None and default not in options:
                    issues.append(SchemaIssue(WARNING, where,
                                              f"default {default!r} is not one of 'options'"))
        elif options is not None:
            issues.append(SchemaIssue(WARNING, where,
                                      f"'options' is only meaningful for type 'select', "
                                      f"not {ptype!r}"))

        # ── bool ───────────────────────────────────────────────────────────
        if ptype == "bool" and "default" in entry and not isinstance(default, bool):
            issues.append(SchemaIssue(WARNING, where,
                                      f"'bool' default should be True/False, got {default!r}"))

        # ── colour ─────────────────────────────────────────────────────────
        if ptype == "colour" and isinstance(default, str):
            ok = (default.startswith("#") and len(default) == 7
                  and all(c in "0123456789abcdefABCDEF" for c in default[1:]))
            if not ok:
                issues.append(SchemaIssue(WARNING, where,
                                          f"'colour' default {default!r} is not a 7-character "
                                          f"#rrggbb string — the colour input expects that form"))

        # ── show_if ────────────────────────────────────────────────────────
        show_if = entry.get("show_if")
        if show_if is not None:
            if not isinstance(show_if, dict) or "key" not in show_if or "values" not in show_if:
                issues.append(SchemaIssue(ERROR, where,
                                          "'show_if' must be {'key': other_param, "
                                          "'values': [...]}"))
            else:
                if not isinstance(show_if["values"], (list, tuple)):
                    issues.append(SchemaIssue(ERROR, where, "'show_if.values' must be a list"))
                if show_if["key"] not in all_keys:
                    issues.append(SchemaIssue(ERROR, where,
                                              f"'show_if.key' {show_if['key']!r} is not a param "
                                              f"in this schema — the param will never be shown"))
                if show_if["key"] == key:
                    issues.append(SchemaIssue(ERROR, where, "'show_if.key' refers to itself"))

        # ── browse ─────────────────────────────────────────────────────────
        browse = entry.get("browse")
        if browse is not None:
            if browse not in BROWSE_KINDS:
                issues.append(SchemaIssue(ERROR, where,
                                          f"'browse' must be one of {sorted(BROWSE_KINDS)}, "
                                          f"got {browse!r}"))
            elif ptype != "text":
                issues.append(SchemaIssue(WARNING, where,
                                          "'browse' only applies to a 'text' param"))

        # ── unknown keys (typo catcher) ────────────────────────────────────
        for unknown in sorted(set(entry) - KNOWN_ENTRY_KEYS):
            issues.append(SchemaIssue(WARNING, where,
                                      f"unrecognised key {unknown!r} — the host app ignores it "
                                      f"silently. Known keys: {sorted(KNOWN_ENTRY_KEYS)}"))

    return issues
