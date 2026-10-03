"""Param descriptors -- declare a node's parameters as class attributes.

Instead of a hand-written ``param_schema`` list of dicts::

    class Brightness(Filter):
        brightness = Slider(0, -150, 150, step=1,
                            help="Shifts the overall brightness.")
        mode = Select("normal", {"normal": "plain", "multiply": "darker"},
                      help="Blend algorithm.")

each attribute is a :class:`Param`; :func:`params_from_class` collects them in
definition order and the kind base classes (``core/nodes/kinds.py``) turn them
into the exact ``param_schema`` list ``BaseNode`` has always validated, so
nothing downstream (the API, the UI, A5 coercion, the snapshot) can tell the two
spellings apart. ``from_schema`` is the inverse, used by the round-trip test and
by migrations.

This module deliberately imports nothing from the app, so it can be vendored
into the node SDK byte for byte (analysis/54 5.13).
"""

from __future__ import annotations

__all__ = [
    "Param", "Slider", "Number", "Bool", "Select", "Colour", "Text", "Raw",
    "from_schema", "params_from_class", "ParamView",
]


class Param:
    """One parameter of a node. Base class; use the typed subclasses."""

    schema_type: str = ""

    def __init__(self, default, *, help, label=None, key=None, show_if=None, **extra):
        if not isinstance(help, str) or not help.strip():
            raise TypeError("a param needs a non-empty `help` sentence (analysis/54 5.4)")
        self.default = default
        self.help = help
        self._label = label
        self._key_override = key
        self.key = key            # filled in by __set_name__ when not given
        self.attr = None          # the class attribute name, when declared on a class
        self.show_if = show_if
        self.extra = extra

    # -- descriptor protocol ------------------------------------------------
    def __set_name__(self, owner, name):
        self.attr = name
        if self._key_override is None:
            self.key = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return instance.params[self.key]

    # -- schema -------------------------------------------------------------
    @property
    def label(self) -> str:
        if self._label is not None:
            return self._label
        return (self.key or "").replace("_", " ").title()

    def _typed(self) -> dict:
        """Fields specific to the subclass, in schema order."""
        return {}

    def _help_text(self) -> str:
        return self.help

    def to_schema(self) -> dict:
        """The dict ``BaseNode.param_schema`` has always held for this param."""
        if not self.key:
            raise TypeError("a param has no key: declare it as a class attribute or pass key=")
        d = {"key": self.key, "label": self.label, "type": self.schema_type,
             "default": self.default}
        d.update(self._typed())
        d["help"] = self._help_text()
        if self.show_if is not None:
            d["show_if"] = self.show_if
        d.update(self.extra)
        return d

    def __repr__(self):
        return f"<{type(self).__name__} {self.key or self.attr or '?'}={self.default!r}>"


class Slider(Param):
    schema_type = "slider"

    def __init__(self, default, min, max, step=None, **kw):
        super().__init__(default, **kw)
        self.min, self.max, self.step = min, max, step

    def _typed(self):
        d = {"min": self.min, "max": self.max}
        if self.step is not None:
            d["step"] = self.step
        return d


class Number(Param):
    schema_type = "number"

    def __init__(self, default, min=None, max=None, step=None, **kw):
        super().__init__(default, **kw)
        self.min, self.max, self.step = min, max, step

    def _typed(self):
        d = {}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.step is not None:
            d["step"] = self.step
        return d


class Bool(Param):
    schema_type = "bool"

    def __init__(self, default=False, **kw):
        super().__init__(default, **kw)


class Select(Param):
    """``options`` is a list of values, or a dict ``{value: description}``.

    The dict form emits the keys as ``options`` and appends one
    ``\\n• <value> — <description>`` bullet per entry to ``help``, in order.
    """

    schema_type = "select"

    def __init__(self, default, options, **kw):
        super().__init__(default, **kw)
        if isinstance(options, dict):
            self.options = list(options)
            self.bullets = dict(options)
        else:
            self.options = list(options)
            self.bullets = None

    def _typed(self):
        return {"options": list(self.options)}

    def _help_text(self):
        if not self.bullets:
            return self.help
        return self.help + "".join(f"\n• {k} — {v}" for k, v in self.bullets.items())


class Colour(Param):
    schema_type = "colour"


class Text(Param):
    """Free text; ``browse=``, ``browse_default_name=``, ``placeholder=`` go through ``**kw``."""

    schema_type = "text"

    def __init__(self, default="", **kw):
        super().__init__(default, **kw)


class Raw(Param):
    """Every other schema type (textarea, curve, range, matrix, hue, wiring,
    swatches, breakpoints, colorwheel, conditions): the type-specific fields go
    through ``**kw`` unchanged."""

    def __init__(self, schema_type, default, **kw):
        super().__init__(default, **kw)
        self.schema_type = schema_type


_COMMON = ("key", "label", "type", "default", "help", "show_if")


def from_schema(entry: dict) -> Param:
    """Inverse of ``Param.to_schema``: the descriptor for one ``param_schema`` entry.

    Always builds a list-form ``Select`` (a bulleted help string is just text
    here). Unknown fields are kept as ``**extra``, so the round trip is exact.
    """
    t = entry.get("type")
    common = {"help": entry["help"], "label": entry.get("label"), "key": entry["key"],
              "show_if": entry.get("show_if")}
    rest = {k: v for k, v in entry.items() if k not in _COMMON}
    default = entry.get("default")
    if t == "slider":
        lo, hi = rest.pop("min"), rest.pop("max")
        return Slider(default, lo, hi, step=rest.pop("step", None), **common, **rest)
    if t == "number":
        return Number(default, min=rest.pop("min", None), max=rest.pop("max", None),
                      step=rest.pop("step", None), **common, **rest)
    if t == "bool":
        return Bool(default, **common, **rest)
    if t == "select":
        return Select(default, rest.pop("options"), **common, **rest)
    if t == "colour":
        return Colour(default, **common, **rest)
    if t == "text":
        return Text(default, **common, **rest)
    return Raw(t, default, **common, **rest)


def _is_explicit_schema(klass) -> bool:
    d = klass.__dict__
    return bool(d.get("param_schema")) and not d.get("_params_derived")


def params_from_class(cls):
    """Collect the :class:`Param` attributes of *cls* into a ``param_schema`` list.

    Returns ``None`` when neither *cls* nor any base declares a descriptor (the
    class keeps whatever ``param_schema`` it inherits or defines).

    * Order is MRO-aware: bases first, each class in definition order. A subclass
      that redeclares an attribute replaces the parameter but keeps its position.
    * The attribute name is the key unless ``key=`` was given. An attribute that
      shadows any other attribute of a base class (``color``, ``label``, a kind's
      ``float_input``, ...) raises ``TypeError``: declare ``fill = Colour(...,
      key="color")`` instead.
    * Descriptors and a hand-written ``param_schema`` in the same class body raise
      ``TypeError``, and so do descriptors declared by a class between *cls* and
      the nearest explicit ``param_schema`` of an ancestor.
    * A subclass whose OWN body sets ``param_schema`` (even ``[]``) replaces the
      descriptors it inherits: the hidden legacy alias with its own schema
      (``color_grade_3way`` over ``color_grading``). ``None`` is returned and the
      explicit schema stays.
    """
    # The nearest class that carries a hand-written schema; descriptors declared
    # further up the MRO than that class are replaced by it, not merged with it.
    boundary = None
    for klass in cls.__mro__:
        d = klass.__dict__
        if "param_schema" in d and not d.get("_params_derived") and (d["param_schema"] or klass is cls):
            boundary = klass
            break
    mro = cls.__mro__
    scope = mro if boundary is None else mro[:mro.index(boundary) + 1]
    found: dict = {}
    for klass in reversed(scope):
        for name, val in klass.__dict__.items():
            if isinstance(val, Param):
                found[name] = val
    if boundary is not None:
        if found:
            raise TypeError(
                f"{cls.__name__}: declares param descriptors ({', '.join(found)}) and also "
                f"has an explicit param_schema -- use one or the other")
        return None
    if not found:
        return None
    for name in found:
        for base in cls.__mro__:
            if name in base.__dict__ and not isinstance(base.__dict__[name], Param):
                raise TypeError(
                    f"{cls.__name__}: param attribute {name!r} shadows an existing "
                    f"attribute of {base.__name__}; give it another attribute name and "
                    f"keep the schema key with `key={name!r}`")
    keys: dict = {}
    out = []
    for name, p in found.items():
        if p.key is None:     # declared on an object that never went through __set_name__
            p.key = name
        if p.key in keys:
            raise TypeError(f"{cls.__name__}: parameters {keys[p.key]!r} and {name!r} "
                            f"share the key {p.key!r}")
        keys[p.key] = name
        out.append(p.to_schema())
    return out


class ParamView:
    """Read-only attribute access to a node's parameters: ``p.angle``.

    Reads go through ``node.params`` (``ExprParams``), so a value is resolved
    (``{{ }}`` templates) and coerced to its schema (A5) exactly like
    ``self.params["angle"]``. Assignment raises: params are read-only while a
    node runs (A6).
    """

    __slots__ = ("_node", "_raw")

    def __init__(self, node, raw=False):
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_raw", raw)

    def _fixed(self):
        """The class's ``fixed_params`` (empty for the un-overlaid view)."""
        if self._raw:
            return {}
        return getattr(type(self._node), "fixed_params", None) or {}

    def _pinned(self, key):
        v = self._fixed()[key]
        return v(ParamView(self._node, raw=True)) if callable(v) else v

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._fixed():
            return self._pinned(name)
        try:
            return self._node.params[name]
        except KeyError:
            raise AttributeError(
                f"{type(self._node).__name__} has no parameter {name!r}") from None

    def __setattr__(self, name, value):
        raise AttributeError("parameters are read-only while a node runs; compute locally")

    def __delattr__(self, name):
        raise AttributeError("parameters are read-only while a node runs")

    def __getitem__(self, key):
        if key in self._fixed():
            return self._pinned(key)
        return self._node.params[key]

    def __contains__(self, key):
        return key in self._fixed() or key in self._node.params

    def get(self, key, default=None):
        if key in self._fixed():
            return self._pinned(key)
        return self._node.params.get(key, default)

    def __repr__(self):
        return f"<ParamView {type(self._node).__name__}>"
