"""
Base node class for the modular node architecture.

Each node type subclasses BaseNode and provides:
  - type: unique identifier string
  - label: display name
  - category: grouping (Detection, Removal, Post-Process, etc.)
  - color: UI accent colour
  - param_schema: list of parameter definitions
  - execute(): the actual processing logic
"""

import difflib
import inspect
import threading

import numpy as np
from enum import Enum

from .expression import ExprParams


# Global node registry: {node_type: node_class}
NODE_REGISTRY: dict[str, type] = {}


def _state_on_self(node, name):
    try:
        from core.node_contract import contract_violation
    except ImportError:   # vendored copy (node SDK) without the app contract
        return
    contract_violation(
        getattr(node, "type", type(node).__name__), "state-on-self",
        f"execute() rebinds self.{name} -- node instances are shared across the "
        f"per-image thread pool (analysis/51 F11); compute locally or keep it in "
        f"context")


def _import_rule(cls, rule, message):
    """An import-time rule added in SDK 1.2: raise in strict mode, warn otherwise.

    Plugins written against 1.1 must still load (additive-only, SDK
    CONTRIBUTING.md), so production logs once and tests/fuzzers raise.
    """
    try:
        import config
        strict = bool(getattr(config, "STRICT_CONTRACT", False))
    except ImportError:   # vendored copy (node SDK) without the app config
        strict = False
    if strict:
        raise TypeError(message)
    try:
        from core.node_contract import contract_violation
    except ImportError:
        import warnings
        warnings.warn(message, stacklevel=3)
        return
    contract_violation(getattr(cls, "type", cls.__name__), rule, message)


def _signature_accepts(fn, name: str) -> bool:
    """Does callable ``fn`` take a parameter called ``name`` (named or via ``**kwargs``)?

    Used by ``BaseNode.__init_subclass__`` (flag/signature agreement, derived
    ``accepts_data``) and by the engine's kwargs dispatch. An uninspectable
    callable counts as accepting, so it can never be rejected on a guess.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if name in params:
        return params[name].kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


# A6 (strict mode only): ids of the nodes this thread is inside execute() of.
# Filled by core.node_contract.executing(); empty -- so every guard below is a
# single failed lookup -- in production.
_EXECUTING = threading.local()


class ComputeBackend(Enum):
    CPU = "cpu"
    GPU = "gpu"


class BaseNode:
    """Abstract base for all pipeline nodes."""

    type = "base"
    label = "Base Node"
    category = "Other"
    color = "#888888"
    description = ""  # one-line description shown in palette and params panel

    # Extra search aliases for the node picker/command palette — words or
    # phrases a user might type that aren't already substrings of `label`,
    # `type`, `category`, or `description`. The main use case: a node was
    # renamed or absorbed another node's job (e.g. Mask Dilate now also
    # covers what used to be a separate "Mask Expand" node), and searching
    # the old name should still surface the node that does that job today.
    # Purely additive — safe default is an empty tuple; never required.
    search_terms: tuple[str, ...] = ()
    # Attribute names execute() may (re)bind on ``self`` despite A6: only
    # lazily-initialised caches guarded by a lock. Mutating a container in
    # place (``self._cache[k] = v``) was never rebinding and needs no entry.
    thread_safe_attrs: tuple[str, ...] = ()
    # How many image edges an accepts_multiple node needs before its
    # execute_multiple() does its real work. Only channel_merge differs from
    # the two-input default; read by core/lint.py and scripts/goldens.py.
    multi_input_count: int = 2
    # Override `multi_input_count` for certain parameter settings, keyed by
    # the count to use and valued like a param's `show_if` (a list of
    # conditions is OR-ed, same as `port_show_if`/`required_inputs_if`).
    # Resolved per node INSTANCE by `resolve_multi_input_count()`.
    #
    # Most accepts_multiple nodes need every slot at every setting (Blend's
    # background/foreground). Assert is the odd one out: its second slot is a
    # REFERENCE edge that only the `matches_reference` test reads, so at every
    # other setting the node needs exactly one — and the frontend drew a
    # second, permanently-unused dot on whichever stream it was checking
    # (image, mask or samples) regardless of which test was picked, which is
    # also what let two edges from the SAME source silently attach there with
    # nothing to compare. Nodes like this declare the narrow count as their
    # baseline `multi_input_count` and widen it here only for the setting that
    # actually reads a second edge.
    multi_input_count_if: dict[int, object] = {}

    # Excluded from the node palette (list_node_types) while remaining fully
    # functional for execution — lets existing saved workflows keep working
    # even after a node is retired from new-workflow use.
    hidden = False

    param_schema: list[dict] = []  # list of param dicts
    supports_gpu = False
    accepts_mask     = False  # orange mask input port
    produces_mask    = False  # orange mask output port
    produces_image   = True   # blue image output port
    has_input_port   = True   # blue image input port
    has_output_port  = True   # blue image output port
    accepts_samples  = False  # red sample input port
    produces_samples = False  # red sample output port
    accepts_data     = False  # green data input port
    # Green data output port. Defaults to True so the data stream stays
    # *continuous*: the engine already carries an incoming data payload past a
    # node that returns no "data" key of its own (core/pipeline.py's
    # `acc_data = out.get("data") ... else carry_data`), but with the flag off
    # neither lint nor the UI would let you draw the outgoing edge, so the
    # stream was severed at the first ordinary image node and `{{ $json.x }}`
    # could not survive a Resize. Nodes that compute real data simply return it
    # under "data" and override nothing. Terminal nodes are the exception:
    # __init_subclass__ forces this back to False on is_sink_node classes,
    # which by definition have no outgoing connections.
    produces_data    = True   # green data output port (pass-through by default)

    # When True, this node's image port accepts more than one incoming edge
    # (e.g. Blend's background + foreground, Displace's source + displacement
    # map) instead of the usual single connection. core/pipeline.py pairs
    # same-port edges up in the order the connections were made and calls
    # execute_multiple(images, masks, *, samples=None, datas=None,
    # context=None) with the full list, instead of execute(image, mask).
    # Define BOTH methods: execute_multiple for the real multi-input case, and
    # execute() as a single-input fallback (typically "pass through the one
    # image unchanged") for when only one edge is connected.
    accepts_multiple = False

    # Normally the image port is auto-marked required (list_node_types()
    # inserts "image" into required_inputs/required_outputs) whenever the
    # node has that port and isn't a source/sink. Set True on nodes where the
    # image connection itself should stay optional — e.g. a routing node that
    # should also work as a pure mask/samples/data pass-through with no image
    # wired in at all.
    optional_image_port = False

    # Which non-image input ports are REQUIRED for the node to function.
    # The image input is auto-marked required whenever `has_input_port` is True
    # and the node is not a source. Override on subclasses to mark mask /
    # samples / data as required (filled port dot in the UI + a lint error if
    # disconnected at runtime). Example: `required_inputs = ("samples",)`.
    required_inputs: tuple[str, ...] = ()

    # OR-groups of input ports: at least ONE port in each group must be wired,
    # but no single one of them is individually mandatory. Use this instead of
    # `required_inputs` when the node accepts either of two ports interchange-
    # ably (e.g. a seed source that can come from a mask OR a sample list).
    # Example: `required_input_groups = (("mask", "samples"),)`. Consumers that
    # only understand `required_inputs` (lint, the fuzzer's graph builder)
    # must also check this so they don't build/allow a graph where every port
    # in a group is left disconnected.
    required_input_groups: tuple[tuple[str, ...], ...] = ()

    # Which output ports the node ALWAYS produces (its primary product). The
    # image output is auto-marked required whenever `produces_image` is True,
    # `has_output_port` is True, and the node is not a sink — i.e. the node
    # actually generates a new image rather than passing one through. Override
    # to mark mask / samples / data outputs as primary products. Example:
    # ``required_outputs = ("mask",)`` for a mask-producing node whose image
    # output is just a pass-through.
    required_outputs: tuple[str, ...] = ()
    # Which ports exist only for certain PARAMETER settings, keyed the same way
    # as `port_docs` ("<in|out>.<port>") and valued like a param's `show_if`:
    # ``{"key": "collect", "values": ["image"]}``. A port not named here is
    # always shown.
    #
    # The topology flags above answer "does this node type have a mask input?".
    # They cannot answer "does it have one AT THIS SETTING?", and several nodes
    # genuinely change shape with a parameter — Accumulate collects exactly one
    # of image/mask/data and the other two can never carry anything. Before
    # this existed the only way to express that was a hand-written special case
    # in the frontend (`inferAccumulateNodeType`), so exactly one node had it
    # and every other node with the same problem showed dead, permanently
    # empty ports on the canvas.
    #
    # Declared here rather than in the frontend so the rule lives next to the
    # `execute()` that makes it true. Shipped through /api/node_types; the UI
    # hides the port, refuses connections to it, and drops any wire that a
    # parameter change just orphaned (see stores/pipelineStore.ts). The ENGINE
    # deliberately does not enforce it: a hidden port that still receives data
    # from an older saved workflow behaves exactly as it did before, so adding
    # a rule can never change how an existing graph runs.
    #
    # A list of conditions is OR-ed, as is `values` within one condition, so
    # the whole thing reads "shown when any of these hold" — which is what the
    # real cases need (Accumulate's Data port is live when collect is data OR
    # when flush is manual). AND has no use case yet and is deliberately not
    # expressible rather than guessed at.
    port_show_if: dict[str, object] = {}
    # Which input ports are REQUIRED only for certain parameter settings, keyed
    # by bare port name ("mask", "samples", "data", "image") and valued like a
    # param's `show_if`: ``{"key": "check", "values": ["mask"]}``. A list of
    # conditions is OR-ed, as in `port_show_if`.
    #
    # `required_inputs` can only say "always" or "never", and a node whose whole
    # job is picking a stream needs "it depends". Assert declared
    # nothing required, so the one port it actually reads showed as an optional
    # hollow dot and an unwired graph passed lint — the failure then arrived at
    # run time as "the mask selects no pixels", which is the node's message for
    # an EMPTY mask, not for a mask that was never connected. Two different
    # mistakes, one indistinguishable error.
    #
    # A port named here must not also appear in `required_inputs`; that would
    # claim it is both always and conditionally required. Resolved per node
    # INSTANCE by `resolve_required_inputs()`; the engine's own runtime guard
    # and `core/lint.py` both go through it, so the rule cannot mean one thing
    # to the linter and another to the runner.
    required_inputs_if: dict[str, object] = {}

    # Opt out of the runner's starvation guard (`IMAGIRA_STRICT_INPUTS`, on by
    # default), which SKIPS a node whose required inputs are absent.
    #
    # For almost every node, skipping is right: nothing useful can be computed
    # from a missing input, and passing None through produces a plausible wrong
    # result further downstream. For an ASSERT it is the worst possible
    # outcome — the guard exists to fail loudly on a missing or empty input,
    # and being skipped means it silently does not guard. `assert_mask_
    # non_empty` had exactly this bug once already (it skipped its own check
    # when no image was wired), and declaring `required_inputs_if` on
    # `assert_node` reintroduced it: the port became required, so the
    # runner skipped the node, so the assertion never ran.
    #
    # Lint still reports the disconnected port — that pre-run signal is the
    # useful half. This flag only stops the RUNNER from quietly standing the
    # node down.
    runs_with_missing_inputs = False
    # Per-port purpose — one short sentence per port, keyed "<in|out>.<port>"
    # (ports: image, mask, samples, data). Answers the question the port dots
    # cannot: what an optional mask input actually does on THIS node, which of
    # two image edges is the background, what a data output carries. Shipped to
    # the UI through /api/node_types and rendered into the generated reference
    # pages by scripts/build_node_docs.py, so one sentence in the node file
    # reaches every place a port is shown.
    # Purely additive — an empty dict is the safe default. A key is only
    # accepted for a port the class actually declares; __init_subclass__ raises
    # at import time otherwise, so the text cannot outlive the port it
    # describes. Not merged across inheritance: like `i18n`, a subclass that
    # adds a port defines its own complete dict.
    port_docs: dict[str, str] = {}
    # i18n — optional per-node translation overrides (see analysis/53_surface.md).
    # First-party translations live in core/nodes/_locales/<locale>.json, keyed by node
    # type; node files carry no i18n. A class-level dict is still supported (plugins) and
    # overrides the file key by key (core/nodes/_i18n_data.py::effective_i18n).
    # Structure: {"de": {"label": "…", "description": "…", "params": {"key": {"label": "…", "help": "…"}}}}
    # English class attributes are the canonical source; this dict only contains overrides.
    # Missing keys fall back silently to the English value. Option *values* are never translated.
    # IMPORTANT: subclasses that provide translations must define their own `i18n = {...}`.
    # Never mutate this dict at runtime — it is a shared class-level object.
    i18n: dict = {}

    # Topology flags — used by app.py and execute.js to avoid hardcoded type strings
    # Does this node START a run — i.e. is it one of the things that decides
    # what images the run is about?
    #
    # Distinct from `is_source_node`, which is only "needs no upstream image".
    # The engine used to derive triggers from port topology alone, which swept
    # in Set Data, Random Seed, Datestamp and Constant Color — nodes that
    # supply a CONSTANT alongside a run rather than starting one. A workflow
    # whose only source is a Set Data has nothing to run over; a workflow with
    # two folder loaders has an ambiguity worth asking about. Those are
    # different questions and the flag separates them.
    #
    # With more than one trigger in a graph, a run has to name the one it
    # means (`run_pipeline_graph(..., trigger_node_id=...)`), because "both"
    # is not an answer anyone intends.
    is_trigger        = False
    is_source_node    = False  # node produces its own images (no upstream image needed)
    is_folder_loader  = False  # source node that batch-loads from a filesystem folder
    is_preview_output = False  # node marks the end of a preview-only pipeline
    is_disk_exporter  = False  # node writes files to disk
    is_sink_node      = False  # node has no output port (terminal)

    # Filled in by __init_subclass__ from each subclass's param_schema.
    _DEFAULT_PARAMS: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Cache defaults once per subclass; cheaper than rebuilding on every
        # __init__ call (matters during batch executions that instantiate
        # many nodes per run).
        cls._DEFAULT_PARAMS = {
            p["key"]: p.get("default") for p in cls.param_schema
        }

        # A sink is terminal — it has no outgoing connections at all — so the
        # pass-through default for `produces_data` must not give it a green
        # output dot. Only override the inherited default; a sink that
        # deliberately spells out `produces_data` in its own body keeps it.
        if getattr(cls, "is_sink_node", False) and "produces_data" not in cls.__dict__:
            cls.produces_data = False

        # Validate required_inputs / required_outputs consistency at class
        # definition time so misconfigured nodes are caught at import, not at
        # runtime.  Only meaningful on concrete subclasses that declare a type.
        if not getattr(cls, "type", "base") or cls.type == "base":
            return

        # Derive `accepts_data` when no class body in the chain declares it
        # (analysis/54 Part 5, A7): every node with parameters can have them
        # bound from the green data port, provided its execute() can take the
        # `data` argument. An explicit declaration -- on this class or any
        # ancestor below BaseNode -- always wins. A value derived for a parent
        # is marked and does not count as explicit for its subclasses.
        if not any(
            "accepts_data" in k.__dict__ and not k.__dict__.get("_accepts_data_derived")
            for k in cls.__mro__ if k is not BaseNode and k is not object
        ):
            cls.accepts_data = bool(cls.param_schema) and _signature_accepts(cls.execute, "data")
            cls._accepts_data_derived = True

        # Category must come from the canonical list (analysis/54 Part 5, A8):
        # a typo would otherwise silently open a new palette group.
        try:
            from config import NODE_CATEGORIES as _CATS
        except ImportError:   # vendored copy (node SDK) without the app config
            _CATS = None
        # Two things are not a typo: the inherited default "Other" (the author
        # set nothing, so there is nothing to correct) and the "Custom" flow,
        # whose categories are chosen by the user in api/custom_nodes.py.
        # First-party nodes only: a marketplace/entry-point plugin may bring its
        # own category, and raising here would drop it from discovery entirely.
        if (_CATS is not None and str(cls.__module__).startswith("core.nodes.")
                and cls.category not in _CATS
                and cls.category != "Other"
                and str(cls.category).split("|")[0] != "Custom"):
            close = difflib.get_close_matches(str(cls.category), _CATS, n=1, cutoff=0.5)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise TypeError(
                f"{cls.__name__}: category {cls.category!r} is not in config.NODE_CATEGORIES."
                f"{hint} A new category needs an entry there and in NodePalette.tsx."
            )

        # Dispatch kwargs follow flag AND signature (analysis/54 Part 5, A1): a
        # stream flag without the matching execute() parameter used to be a
        # TypeError on the first run; now it is an error at import.
        for _flag, _arg in (("accepts_samples", "samples"),
                            ("accepts_data", "data"),
                            ("accepts_context", "context")):
            if getattr(cls, _flag, False) and not _signature_accepts(cls.execute, _arg):
                _import_rule(
                    cls, "flag-without-param",
                    f"{cls.__name__}: {_flag} is True but execute() has no `{_arg}` "
                    f"parameter (and no **kwargs) -- the engine would pass `{_arg}=` "
                    f"and crash. Add `{_arg}=None` to execute() or drop the flag."
                )
        _PORT_ACCEPTS = {
            "mask":    "accepts_mask",
            "samples": "accepts_samples",
            "data":    "accepts_data",
            "image":   "has_input_port",
        }
        for port in cls.required_inputs:
            flag = _PORT_ACCEPTS.get(port)
            if flag and not getattr(cls, flag, False):
                raise TypeError(
                    f"{cls.__name__}: required_inputs includes {port!r} but "
                    f"{flag} is False — a required port must also be accepted."
                )
        for group in cls.required_input_groups:
            if len(group) < 2:
                raise TypeError(
                    f"{cls.__name__}: required_input_groups entry {group!r} needs "
                    f"at least 2 ports — a single-port group belongs in required_inputs."
                )
            for port in group:
                flag = _PORT_ACCEPTS.get(port)
                if flag and not getattr(cls, flag, False):
                    raise TypeError(
                        f"{cls.__name__}: required_input_groups includes {port!r} but "
                        f"{flag} is False — a required port must also be accepted."
                    )
        _PORT_PRODUCES = {
            "mask":    "produces_mask",
            "samples": "produces_samples",
            "data":    "produces_data",
            "image":   "has_output_port",
        }
        for port in cls.required_outputs:
            flag = _PORT_PRODUCES.get(port)
            if flag and not getattr(cls, flag, False):
                raise TypeError(
                    f"{cls.__name__}: required_outputs includes {port!r} but "
                    f"{flag} is False — a required output port must also be produced."
                )
        _PORT_DOC_FLAGS = {
            "in.image":    "has_input_port",
            "in.mask":     "accepts_mask",
            "in.samples":  "accepts_samples",
            "in.data":     "accepts_data",
            "out.image":   "has_output_port",
            "out.mask":    "produces_mask",
            "out.samples": "produces_samples",
            "out.data":    "produces_data",
        }
        for port, cond in cls.required_inputs_if.items():
            flag = _PORT_ACCEPTS.get(port)
            if flag is None:
                raise TypeError(
                    f"{cls.__name__}: required_inputs_if key {port!r} is not an input port "
                    f"— expected one of {', '.join(sorted(_PORT_ACCEPTS))}."
                )
            if not getattr(cls, flag, False):
                raise TypeError(
                    f"{cls.__name__}: required_inputs_if names {port!r} but {flag} is "
                    f"False — a required port must also be accepted."
                )
            if port in cls.required_inputs:
                raise TypeError(
                    f"{cls.__name__}: {port!r} is in both required_inputs and "
                    f"required_inputs_if — it cannot be both always and conditionally "
                    f"required. Pick one."
                )
            _cond_param_keys = {p["key"] for p in cls.param_schema}
            for c in (cond if isinstance(cond, (list, tuple)) else [cond]):
                if not isinstance(c, dict) or "key" not in c or "values" not in c:
                    raise TypeError(
                        f"{cls.__name__}: required_inputs_if[{port!r}] must be a "
                        f"{{'key': ..., 'values': [...]}} dict (or a list of them), got {c!r}."
                    )
                if c["key"] not in _cond_param_keys:
                    raise TypeError(
                        f"{cls.__name__}: required_inputs_if[{port!r}] refers to parameter "
                        f"{c['key']!r}, which is not in param_schema — the port could never "
                        f"become required."
                    )

        for candidate, cond in cls.multi_input_count_if.items():
            if not getattr(cls, "accepts_multiple", False):
                raise TypeError(
                    f"{cls.__name__}: declares multi_input_count_if but "
                    f"accepts_multiple is False — there is no multi-input port "
                    f"to widen."
                )
            if not isinstance(candidate, int) or candidate <= cls.multi_input_count:
                raise TypeError(
                    f"{cls.__name__}: multi_input_count_if key {candidate!r} must be "
                    f"an int greater than the baseline multi_input_count "
                    f"({cls.multi_input_count!r}) — it widens the slot count, it "
                    f"cannot narrow or repeat it."
                )
            _cond_param_keys = {p["key"] for p in cls.param_schema}
            for c in (cond if isinstance(cond, (list, tuple)) else [cond]):
                if not isinstance(c, dict) or "key" not in c or "values" not in c:
                    raise TypeError(
                        f"{cls.__name__}: multi_input_count_if[{candidate!r}] must be a "
                        f"{{'key': ..., 'values': [...]}} dict (or a list of them), got {c!r}."
                    )
                if c["key"] not in _cond_param_keys:
                    raise TypeError(
                        f"{cls.__name__}: multi_input_count_if[{candidate!r}] refers to "
                        f"parameter {c['key']!r}, which is not in param_schema — the slot "
                        f"could never widen."
                    )

        # A port cannot be unconditionally required AND sometimes hidden: the
        # graph would be unsatisfiable at the settings that hide it, with the
        # UI refusing the very connection the linter demands.
        for port in cls.required_inputs:
            if f"in.{port}" in cls.port_show_if:
                raise TypeError(
                    f"{cls.__name__}: {port!r} is in required_inputs but its port is "
                    f"conditionally hidden by port_show_if — at the settings that hide "
                    f"it the graph could never satisfy the requirement. Move it to "
                    f"required_inputs_if with a matching condition."
                )

        for key, cond in cls.port_show_if.items():
            flag = _PORT_DOC_FLAGS.get(key)
            if flag is None:
                raise TypeError(
                    f"{cls.__name__}: port_show_if key {key!r} is not a port — expected "
                    f"one of {', '.join(sorted(_PORT_DOC_FLAGS))}."
                )
            if not getattr(cls, flag, False):
                raise TypeError(
                    f"{cls.__name__}: port_show_if conditions {key!r} but {flag} is False "
                    f"— the node has no such port to hide."
                )
            param_keys = {p["key"] for p in cls.param_schema}
            for c in (cond if isinstance(cond, (list, tuple)) else [cond]):
                if not isinstance(c, dict) or "key" not in c or "values" not in c:
                    raise TypeError(
                        f"{cls.__name__}: port_show_if[{key!r}] must be a "
                        f"{{'key': ..., 'values': [...]}} dict (or a list of them), got {c!r}."
                    )
                if c["key"] not in param_keys:
                    raise TypeError(
                        f"{cls.__name__}: port_show_if[{key!r}] refers to parameter "
                        f"{c['key']!r}, which is not in param_schema — a condition on a "
                        f"parameter that does not exist would hide the port forever."
                    )
        # Parameter schema integrity. A `type` the UI has no control for does
        # not fail loudly anywhere — ParamInput.tsx's switch falls through to
        # its `default`, so the parameter silently renders as a free-text box.
        # That is how `"type": "int"`, `"color"`, `"image"` and `"textarea"`
        # ended up shipping: a number param with no stepper, a colour param
        # with no swatch, and two params for an image that cannot be typed at
        # all. Nothing was broken enough to notice, and every one of them was
        # a worse control than the author had written. So: the canonical list
        # lives here, next to the schema it describes, and an unknown type is
        # an import-time error rather than a downgraded widget.
        _PARAM_TYPES = {
            "slider", "number", "select", "bool", "text", "textarea",
            "colour", "curve", "breakpoints", "swatches", "colorwheel",
            "matrix", "range", "hue", "wiring", "conditions",
        }
        param_keys = {p["key"] for p in cls.param_schema}
        param_pos = {p["key"]: i for i, p in enumerate(cls.param_schema)}
        for spec in cls.param_schema:
            ptype = spec.get("type")
            if ptype not in _PARAM_TYPES:
                raise TypeError(
                    f"{cls.__name__}: parameter {spec.get('key')!r} has type "
                    f"{ptype!r}, which no UI control renders — it would fall "
                    f"through to a plain text box. Expected one of "
                    f"{', '.join(sorted(_PARAM_TYPES))}."
                )
            # `select`-family params: an option_labels/option_previews entry
            # for a value that is not in `options` is dead text — most often
            # the leftover of a renamed option, which then shows the raw
            # value in the dropdown instead of the label the author wrote.
            options = spec.get("options")
            if ptype in ("select", "swatches"):
                if not options:
                    raise TypeError(
                        f"{cls.__name__}: parameter {spec.get('key')!r} is a "
                        f"{ptype} but declares no `options`."
                    )
                for extra in ("option_labels", "option_previews"):
                    for value in (spec.get(extra) or {}):
                        if value not in options:
                            raise TypeError(
                                f"{cls.__name__}: {extra} for parameter "
                                f"{spec.get('key')!r} names {value!r}, which is "
                                f"not one of its options — a stale entry never "
                                f"reaches the UI."
                            )
            # A `matrix` is a VIEW over sibling numeric params: its cells own
            # the real keys, built from `key_template`. Every cell it claims
            # must exist, or the grid would silently write parameters the node
            # never reads — the failure mode is a control that looks like it
            # works and changes nothing.
            if ptype == "matrix":
                m = spec.get("matrix")
                if not isinstance(m, dict):
                    raise TypeError(
                        f"{cls.__name__}: parameter {spec.get('key')!r} is a matrix but "
                        f"has no `matrix` layout dict."
                    )
                for field in ("rows", "cols", "key_template"):
                    if not m.get(field):
                        raise TypeError(
                            f"{cls.__name__}: matrix {spec.get('key')!r} is missing "
                            f"`{field}`."
                        )
                missing = [
                    m["key_template"].replace("{row}", r).replace("{col}", c)
                    for r in m["rows"] for c in m["cols"]
                    if m["key_template"].replace("{row}", r).replace("{col}", c)
                    not in param_keys
                ]
                if missing:
                    raise TypeError(
                        f"{cls.__name__}: matrix {spec.get('key')!r} names cell "
                        f"parameter(s) {missing[:4]} that are not in param_schema — "
                        f"the grid would write values the node never reads."
                    )

            # `range` is a VIEW over a sibling min/max pair, exactly as `matrix`
            # is over a grid: both ends must exist or the slider would write
            # parameters the node never reads.
            if ptype == "range":
                rng = spec.get("range")
                if not isinstance(rng, dict):
                    raise TypeError(
                        f"{cls.__name__}: parameter {spec.get('key')!r} is a range but has "
                        f"no `range` layout dict."
                    )
                for field in ("low_key", "high_key", "min", "max"):
                    if rng.get(field) is None:
                        raise TypeError(
                            f"{cls.__name__}: range {spec.get('key')!r} is missing `{field}`."
                        )
                for field in ("low_key", "high_key"):
                    if rng[field] not in param_keys:
                        raise TypeError(
                            f"{cls.__name__}: range {spec.get('key')!r} names {rng[field]!r}, "
                            f"which is not in param_schema — the slider would write a value "
                            f"the node never reads."
                        )

            # A `hue` param may point at a sibling tolerance param to draw as
            # an arc; naming one that does not exist would draw nothing and
            # look like a broken dial.
            if ptype == "hue":
                band = (spec.get("hue") or {}).get("band_key")
                if band is not None and band not in param_keys:
                    raise TypeError(
                        f"{cls.__name__}: hue {spec.get('key')!r} names band_key {band!r}, "
                        f"which is not in param_schema."
                    )

            # A wiring diagram whose edges name boxes it does not declare
            # would silently draw nothing — the one failure mode that leaves
            # the user with less than the prose they had before.
            if ptype == "wiring":
                w = spec.get("wiring")
                if not isinstance(w, dict) or not w.get("nodes"):
                    raise TypeError(
                        f"{cls.__name__}: parameter {spec.get('key')!r} is a wiring "
                        f"diagram but declares no nodes."
                    )
                ids = {n.get("id") for n in w["nodes"]}
                for edge in w.get("edges") or []:
                    for end in ("from", "to"):
                        if edge.get(end) not in ids:
                            raise TypeError(
                                f"{cls.__name__}: wiring edge {end}={edge.get(end)!r} names "
                                f"a box that is not in `nodes` — the edge would not be drawn."
                            )

            # A `show_if` on a parameter that does not exist hides the control
            # forever; the same rule already guards port_show_if above.
            for cond in (spec.get("show_if") or []) if isinstance(spec.get("show_if"), (list, tuple)) \
                    else ([spec["show_if"]] if spec.get("show_if") else []):
                if not isinstance(cond, dict) or "key" not in cond or "values" not in cond:
                    raise TypeError(
                        f"{cls.__name__}: show_if on {spec.get('key')!r} must be a "
                        f"{{'key': ..., 'values': [...]}} dict, got {cond!r}."
                    )
                if cond["key"] not in param_keys:
                    raise TypeError(
                        f"{cls.__name__}: show_if on {spec.get('key')!r} refers to "
                        f"parameter {cond['key']!r}, which is not in param_schema "
                        f"— the control would never be shown."
                    )
                if param_pos[cond["key"]] >= param_pos.get(spec.get("key"), 0):
                    _import_rule(
                        cls, "show-if-order",
                        f"{cls.__name__}: show_if on {spec.get('key')!r} gates on "
                        f"{cond['key']!r}, which comes later in param_schema -- the "
                        f"gating parameter must be listed before the one it shows or hides."
                    )

        for key in cls.port_docs:
            flag = _PORT_DOC_FLAGS.get(key)
            if flag is None:
                raise TypeError(
                    f"{cls.__name__}: port_docs key {key!r} is not a port — expected "
                    f"one of {', '.join(sorted(_PORT_DOC_FLAGS))}."
                )
            if not getattr(cls, flag, False):
                raise TypeError(
                    f"{cls.__name__}: port_docs documents {key!r} but {flag} is False "
                    f"— the node has no such port."
                )

    @classmethod
    def _get_param_coercer(cls):
        """Per-class schema coercer for reads (A5), built on first use and
        cached on the class. None when unavailable (vendored SDK copy)."""
        try:
            return cls.__dict__["_param_coercer"]
        except KeyError:
            pass
        try:
            from core.node_contract import build_param_coercer
            co = build_param_coercer(cls)
        except ImportError:   # vendored copy (node SDK) without the app contract
            co = None
        cls._param_coercer = co
        return co

    def __setattr__(self, name, value):
        # A6: one node instance serves every image of the per-image thread
        # pool (analysis/51 F11), so rebinding self.x in execute() leaks state
        # between images. Strict mode only; otherwise the id set is empty.
        ids = _EXECUTING.__dict__.get("ids")
        if ids and id(self) in ids and name not in self.thread_safe_attrs:
            _state_on_self(self, name)
        object.__setattr__(self, name, value)

    def _param_write_guard(self, key):
        """ExprParams write hook (A6): ``self.params`` is read-only in execute()."""
        ids = _EXECUTING.__dict__.get("ids")
        if ids and id(self) in ids:
            try:
                from core.node_contract import contract_violation
            except ImportError:   # vendored copy (node SDK) without the app contract
                return
            contract_violation(
                getattr(self, "type", type(self).__name__), "param-write",
                f"execute() writes self.params[{key!r}] -- params are read-only "
                f"while a node runs (one instance serves every image); compute "
                f"locally")

    def __init__(self, node_id, params=None):
        self.node_id = node_id
        # ExprParams (not a plain dict) transparently resolves n8n-style
        # `{{ $json.path }}` templates on read against whatever data the
        # current execute() call is bound to — see core/expression.py.
        self.params = ExprParams(self._DEFAULT_PARAMS, coerce=self._get_param_coercer())
        if params:
            self.params.update(params)
        self.params._write_guard = self._param_write_guard
        self.visible = True
        self.compute_backend = ComputeBackend.CPU
        # Per-node free-text annotation ("why this node, why these settings").
        # Persisted in the workflow JSON so it travels with the graph.
        self.note: str = ""
        # User-chosen name for THIS node ("Remove Logo" on a mask_delete), as
        # distinct from `label`, which names the node TYPE and is fixed per
        # class. Empty means "show the type's label". The type name is still
        # shown on the card underneath, so a renamed node never hides what it
        # actually is.
        self.custom_label: str = ""

    def _default_params(self):
        """Return default values for all schema parameters.

        Retained for backwards compatibility with any external callers; prefer
        the class-level ``_DEFAULT_PARAMS`` constant.
        """
        return dict(self._DEFAULT_PARAMS)

    def execute(self, image, mask=None, data=None, context=None):
        """Execute the node's processing logic.

        Parameters
        ----------
        image : np.ndarray
            uint8 RGB image (H, W, 3).
        mask : np.ndarray or None
            Boolean mask (H, W), optional.
        data : dict or None
            Structured data from an upstream data-port connection (green).
            Keys are producer-defined (e.g., {"color": [r,g,b], "value": 0.5}).
        context : dict
            Shared state across nodes (e.g., pipeline metadata).

        Returns
        -------
        dict
            {"image": np.uint8, "mask": np.bool_ or None, "data": dict or None}
        """
        raise NotImplementedError

    def execute_gpu(self, image, mask=None, **kwargs):
        """GPU-accelerated execution path.

        The pipeline calls this instead of ``execute()`` when the run is
        dispatched to the GPU backend (``compute_backend="gpu"``) **and**
        ``supports_gpu = True`` is set on the node class.

        Default implementation falls back to ``execute()`` so nodes without a
        GPU path degrade gracefully.  Override in nodes that set
        ``supports_gpu = True``.

        Inputs and outputs are always CPU numpy arrays — GPU transfer is the
        node's own responsibility (see ``core.gpu.to_gpu`` / ``to_cpu``).
        """
        return self.execute(image, mask=mask, **kwargs)

    def get_param_value(self, key):
        return self.params.get(key)

    def set_param_value(self, key, value):
        self.params[key] = value

    @classmethod
    def condition_holds(cls, cond, params) -> bool:
        """Does a `show_if`-shaped condition (or OR-list of them) hold for `params`?

        Values are compared as strings so a select storing `"5"` and a saved
        workflow carrying `5` are the same answer — the same coercion the
        frontend's `portIsVisible` does, kept here so the two cannot disagree
        about whether a port is live.
        """
        for one in (cond if isinstance(cond, (list, tuple)) else [cond]):
            if not isinstance(one, dict):
                continue
            current = (params or {}).get(one.get("key"))
            if current is None:
                current = cls._DEFAULT_PARAMS.get(one.get("key"))
            wanted = one.get("values") or []
            if any(str(current) == str(v) for v in wanted):
                return True
        return False

    @classmethod
    def resolve_required_inputs(cls, params=None) -> tuple:
        """`required_inputs` plus whichever `required_inputs_if` ports apply.

        The one place the conditional form is evaluated. `core/lint.py` and the
        runner's own guard both call this, so a port cannot be required
        according to the linter and optional according to the engine.
        """
        required = list(cls.required_inputs or ())
        for port, cond in (cls.required_inputs_if or {}).items():
            if port not in required and cls.condition_holds(cond, params):
                required.append(port)
        return tuple(required)

    @classmethod
    def resolve_multi_input_count(cls, params=None) -> int:
        """`multi_input_count`, widened by whichever `multi_input_count_if` holds.

        The one place the conditional form is evaluated — /api/node_types
        resolves it per node instance the same way it resolves
        `required_inputs_if`, so the frontend draws exactly as many extra
        slots as the current settings actually use.
        """
        # `or 2` would silently turn an explicit `multi_input_count = 0` into
        # 2 — 0 is falsy but a legitimate value a node class could declare.
        count = int(cls.multi_input_count if cls.multi_input_count is not None else 2)
        for candidate, cond in (cls.multi_input_count_if or {}).items():
            if cls.condition_holds(cond, params):
                count = max(count, int(candidate))
        return count

    def to_dict(self):
        d = {
            "id": self.node_id,
            "type": self.type,
            "label": self.label,
            # Raw values, not resolved: `dict(self.params)`/`**self.params`
            # would go through ExprParams.__getitem__ via the mapping
            # protocol and serialize whatever a `{{ }}` template last
            # resolved to instead of the template text itself. `dict.items`
            # reads the underlying storage directly, bypassing the override.
            "params": dict(dict.items(self.params)),
            "visible": self.visible,
            "compute_backend": self.compute_backend.value,
        }
        if self.note:
            d["note"] = self.note
        if self.custom_label:
            d["custom_label"] = self.custom_label
        return d

    @classmethod
    def from_dict(cls, data):
        node = cls(
            node_id=data["id"],
            params=data.get("params", {}),
        )
        if "visible" in data:
            node.visible = bool(data.get("visible", True))
        cb = data.get("compute_backend")
        if cb:
            try:
                node.compute_backend = ComputeBackend(cb)
            except Exception:
                pass
        custom_label = data.get("custom_label")
        if isinstance(custom_label, str):
            node.custom_label = custom_label
        note = data.get("note")
        if isinstance(note, str):
            node.note = note
        return node

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.node_id}>"


class UnknownNodeType(BaseNode):
    """Placeholder standing in for a node whose type isn't in ``NODE_REGISTRY``.

    A workflow can reference a type that used to exist (renamed, consolidated
    into another node, or belongs to an uninstalled marketplace plugin) —
    ``get_node_class()`` then returns ``None``. ``Pipeline.from_dict`` used to
    respond by silently dropping the node from ``pl.nodes`` entirely, while
    leaving ``pl.connections`` untouched — every edge to or from it still
    named an id that, as far as ``pl.nodes`` was concerned, had never existed.
    That mismatch reached ``run_pipeline_graph``'s topological sort, which
    builds ``in_deg``/``node_depth`` only for ids present in ``nodes`` but
    walks successors from raw connections: a dangling successor id raised a
    bare, uninformative ``KeyError('<the missing node's id>')`` — "Pipeline
    crashed: 'node_xxx'" with no indication *why*.

    Standing in with a real (if inert) node object instead keeps every id
    that ever appeared in ``nodes`` present in ``nodes`` — so the topo-sort,
    lint, and every other id-keyed structure stay internally consistent —
    while `type`/`label`/`params` are preserved byte-for-byte from the saved
    file. That preservation matters twice over: a user who reinstalls the
    missing plugin (or whose in-progress refactor finishes) gets their exact
    node back on the next load, and `_rule_unknown_node_type` (core/lint.py)
    can surface a clear, actionable warning instead of the node simply having
    vanished with no explanation.

    Execution still fails for this node specifically — there is no processing
    logic to fall back to — but it now fails through the SAME "unknown node
    type" placeholder path every other run already uses (run_pipeline_graph
    re-resolves `get_node_class` from the raw type string and hits the
    existing `if not node:` branch), not a crash that takes the whole graph
    down with it.
    """

    label = "Unknown Node"
    category = "Other"
    color = "#e74c3c"
    has_input_port = False
    has_output_port = False
    accepts_mask = False
    produces_mask = False
    produces_image = False

    def __init__(self, node_id, params=None, *, original_type="unknown", original_label=None):
        super().__init__(node_id, params)
        # Instance attributes deliberately shadow the class-level ones above:
        # every OTHER node's `type`/`label` are fixed per-class, but this one
        # stands in for whatever the saved file actually said, so `to_dict()`
        # round-trips it byte-for-byte rather than overwriting it with a
        # generic label the next time the workflow is saved.
        self.type = original_type
        self.label = original_label or f"Unknown ({original_type})"

    @classmethod
    def from_dict(cls, data):
        node = cls(
            node_id=data["id"],
            params=data.get("params", {}),
            original_type=data.get("type", "unknown"),
            original_label=data.get("label"),
        )
        if "visible" in data:
            node.visible = bool(data.get("visible", True))
        custom_label = data.get("custom_label")
        if isinstance(custom_label, str):
            node.custom_label = custom_label
        note = data.get("note")
        if isinstance(note, str):
            node.note = note
        return node

    def execute(self, image, mask=None, data=None, context=None):
        raise RuntimeError(
            f"'{self.type}' is not a registered node type — it may have been "
            f"renamed, consolidated into another node, or belongs to an "
            f"uninstalled plugin. Replace or delete this node."
        )


def register(cls):
    """Register a node class in the global NODE_REGISTRY."""
    previous = NODE_REGISTRY.get(cls.type)
    if previous is not None and previous is not cls:
        # Last registration still wins (unchanged), but a plugin silently
        # replacing a built-in for every saved workflow must be visible.
        import logging
        logging.getLogger(__name__).warning(
            "node type %r re-registered: %s replaces %s",
            cls.type, getattr(cls, "__qualname__", cls), getattr(previous, "__qualname__", previous))
    NODE_REGISTRY[cls.type] = cls
    return cls


def to_rgb(image):
    """Return a guaranteed (H, W, 3) uint8 RGB view of *image*.

    Handles:
    - RGBA (H,W,4) → drop alpha, return first 3 channels
    - Grayscale (H,W) or (H,W,1) → broadcast to 3-channel
    - Already (H,W,3) → returned unchanged (no copy)
    - float32/float16 → clipped and cast to uint8
    """
    if image is None:
        return None
    if image.dtype != np.uint8:
        image = to_uint8(image)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.concatenate([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image


def to_uint8(array):
    """Clamp a float image to 0-255 and ROUND it to uint8.

    The one place in the codebase that decides how a float becomes a pixel.

    `np.clip(x, 0, 255).astype(np.uint8)` — the pattern this replaces, which
    stood at ~150 sites — truncates, because that is what a float-to-int cast
    does in C and therefore in numpy. Every one of those sites lost up to a
    full level on every pixel, always downward, so a chain of ten nodes could
    darken an image by several levels for no reason anyone could point at.

    Worse, it made "neutral" settings not neutral. Selective Color with all 36
    inks at zero is the exact identity in float (`c = 1 - r`, `out = 1 - c =
    r`) — but 220/255*255 lands on 219.99999999999997, so an untouched node
    darkened most of the frame by one level. Barrel / Pincushion had the same
    bug in coordinate space, where truncation shifted half the pixels
    up-and-left at a strength documented as leaving the image unchanged.

    Uses `np.rint` — round-half-to-EVEN — on purpose, and this is deliberately
    the opposite choice from `SamplePointsNode._round_half_up`. A centroid is
    one number whose error is visible as a direction, so it must not have a
    systematic bias toward the top-left. A pixel is one of millions, and what
    matters is that the population is not biased: half-to-even sends exact .5
    up half the time, where half-up would add a small constant brightening to
    every image.

    Accepts any array-like; returns None for None, so it drops into an
    early-exit path without a guard.
    """
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def to_rgb_keep_alpha(image):
    """Like :func:`to_rgb`, but an RGBA input keeps its alpha channel.

    `to_rgb` exists to guarantee three channels so colour maths can index
    `[..., 0:3]` safely, and dropping alpha is the right call there — the
    alpha is put back by `mask_blend` afterwards. Geometry nodes are the
    opposite case: they move pixels rather than recolour them, the alpha has
    to move with them, and there is no later step that could restore it
    because the pixels are no longer where they were. Those nodes still need
    the grayscale-to-3-channel and dtype normalisation, just not the strip.

    Returns (H, W, 3) for grayscale/RGB input and (H, W, 4) for RGBA.
    """
    if image is None:
        return None
    if image.dtype != np.uint8:
        image = to_uint8(image)
    if image.ndim == 3 and image.shape[2] == 4:
        return image
    return to_rgb(image)


def _mask_peak(mask) -> float:
    """The value that means "fully selected" for this mask's dtype.

    uint8 is the awkward one: NODE_RULES.md's input contract allows 0/1 AND 0/255,
    and assuming 255 made a 0/1 mask look SOFT — 1 is strictly between 0 and
    255 — so `mask_is_soft` said yes, `mask_to_uint8` handed PIL a picture of
    almost-black, and `mask_from_uint8` divided by 255 on the way back. A
    perfectly ordinary binary mask came out of every Resize, Rotate and Scale
    To Fit as a float matte of strength 0.0039: the right region selected at
    0.4% strength, which reads as "the effect did almost nothing" rather than
    as an error. Deciding the peak by inspection is what makes both readings
    of uint8 work.
    """
    if np.issubdtype(mask.dtype, np.floating):
        # NODE_RULES.md's contract guarantees a float mask arrives pre-normalized
        # to [0, 1], but `mask_to_uint8`'s own float branch (below) doesn't
        # trust that blindly — it computes `peak = float(mask.max())` in case
        # one isn't. This used to hardcode 1.0 regardless, so an
        # out-of-range float mask fed to `mask_to_float`/`mask_is_soft`
        # clamped everything above 1 to 1.0 (destroying a matte's gradient)
        # while `mask_to_uint8` handled the very same input correctly.
        return max(1.0, float(mask.max())) if mask.size else 1.0
    return 1.0 if (mask.size and int(mask.max()) <= 1) else 255.0


def mask_is_soft(mask) -> bool:
    """True when *mask* carries values strictly between empty and full.

    A soft matte (Mask Feather's output, an alpha) means something a binary
    mask cannot, and a node that is only *transporting* the mask — resizing,
    rotating, compositing — must not flatten it on the way through. Nodes that
    use the mask as a *selection* still binarise with `as_bool_mask`; this is
    for the ones that hand it onward.
    """
    if mask is None:
        return False
    if mask.dtype == np.bool_:
        return False
    peak = _mask_peak(mask)
    return bool(np.any((mask > 0) & (mask < peak)))


def mask_to_uint8(mask):
    """A mask as uint8 0-255, ready to hand to PIL for a resample."""
    if mask is None:
        return None
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255
    if np.issubdtype(mask.dtype, np.floating):
        peak = float(mask.max()) if mask.size else 0.0
        scaled = mask if peak > 1.0 else mask * 255.0
        return to_uint8(scaled)
    # A uint8 0/1 mask has to be scaled up like any other full-strength mask.
    # Returned as-is it became a near-black image for PIL to resample — see
    # `_mask_peak`.
    if _mask_peak(mask) <= 1.0:
        return mask.astype(np.uint8) * 255
    return mask.astype(np.uint8)


def mask_from_uint8(arr, soft: bool):
    """Inverse of :func:`mask_to_uint8` after a resample.

    ``soft`` decides the destination type, and it must come from the mask
    *before* the resample — thresholding at 127 is what silently turned a
    feathered matte back into a hard edge every time it passed a Resize.
    """
    if arr is None:
        return None
    return (arr.astype(np.float32) / 255.0) if soft else (arr > 127)

def mask_to_float(mask):
    """A mask as float32 in [0, 1], whichever of the legal dtypes it arrived as.

    For blending, where the mask is a WEIGHT rather than a selection. Three
    nodes hand-rolled this and the third got it wrong: `halo_suppress` blurred
    `mask.astype(np.float32)` and clipped to [0, 1], which for a uint8 0/255
    mask pins the entire feather ramp at 1.0 and produces the hard edge the
    node exists to avoid. Use `as_bool_mask` for "which pixels", this for "how
    much".
    """
    if mask is None:
        return None
    if mask.dtype == np.bool_:
        return mask.astype(np.float32)
    arr = np.asarray(mask, dtype=np.float32)
    peak = _mask_peak(mask)
    return np.clip(arr / peak, 0.0, 1.0)


def as_bool_mask(mask):
    """Return a strict boolean view of *mask*, or None.

    Masks legitimately arrive as ``bool``, ``uint8`` (0/1 or 0/255), or a float
    soft matte in [0, 1] — see §"The three legal dtypes, and the helper for each job" in NODE_RULES.md. Only ``bool``
    works as a *selection*: ``image[mask]`` with a uint8 mask is integer
    indexing, so it silently returns the wrong pixels or dies with
    ``index 255 is out of bounds for axis 0 with size 96``. Any node that uses
    the mask to index, or with a bitwise operator (``~ & |``), must convert
    first.

    A float matte is thresholded at 0.5 — the pixels the matte covers more than
    it doesn't. Use the raw float (via ``mask_blend``) when you want the soft
    edge instead of a hard selection.
    """
    if mask is None:
        return None
    if mask.dtype == np.bool_:
        return mask
    if np.issubdtype(mask.dtype, np.floating):
        return mask > 0.5
    return mask > 0


def mask_coverage_region(mask):
    """Every pixel the matte touches at all, as a strict boolean.

    The counterpart to :func:`as_bool_mask`, which thresholds a soft matte at
    0.5 — the right rule when the boolean answers "which pixels count as
    SELECTED?" (statistics, island tests, indexing).

    This answers a different question: "which pixels must be REBUILT?" For a
    node that reconstructs pixels — Propagate Fill, Radiant Fill — partial
    coverage still means rebuild me; the matte's value then decides how much
    of the rebuild survives the composite. Thresholding at 0.5 there left the
    outer half of every feathered edge unrepaired AND then blended that
    unrepaired pixel back at its matte weight, which is a no-op: the feather
    did nothing.
    """
    if mask is None:
        return None
    if mask.dtype == np.bool_:
        return mask
    return mask > 0


def restrict_mask_selection(result, mask, node_type):
    """Exclude everything outside *mask* from a Mask|Source node's own
    computed selection — the connected mask input means "only consider this
    region," so a pixel outside it can never end up selected in the output,
    no matter what the node's own thresholding/segmentation decided for it.

    *result* is the node's raw output before the final ``.astype(np.uint8)``
    cast — a boolean selection array or an integer label array (e.g.
    watershed's region IDs) both work, since excluding a pixel just means
    zeroing it either way. Returns *result* unchanged if *mask* is None.
    Mirrors the convention kmeans_cluster.py already established for this
    same node category — see its ``keep`` mask.
    """
    if mask is None:
        return result
    mask_bool = as_bool_mask(mask)
    if mask_bool.shape[:2] != result.shape[:2]:
        rh, rw = result.shape[:2]
        mh, mw = mask_bool.shape[:2]
        raise ValueError(
            f"{node_type}: shape mismatch between the computed result "
            f"{rw}x{rh} and mask {mw}x{mh}."
        )
    # `np.where(mask_bool, result, 0)` promotes a *boolean* result to int64,
    # because the literal 0 is a Python int and numpy resolves the common type.
    # Callers that cast to uint8 afterwards never noticed; a caller that returns
    # the mask directly shipped an int64 mask, which the engine's mask contract
    # rejects. Zeroing a pixel must not change what kind of array this is.
    return np.where(mask_bool, result, np.zeros((), dtype=result.dtype))


def restore_alpha(orig, result):
    """Give *result* back the alpha channel *orig* had, if it lost one.

    Only ever ADDS the original alpha onto a result that has none — a node
    that deliberately computed its own alpha (Remove Masked, Isolate Masked)
    returns 4 channels already and is passed through untouched. A spatial
    mismatch means the node resized the image, in which case the old alpha no
    longer describes it and is dropped rather than misaligned.
    """
    if orig is None or result is None:
        return result
    o = np.asarray(orig)
    r = np.asarray(result)
    if o.ndim != 3 or o.shape[2] != 4:
        return result
    if r.ndim == 2:
        r = r[:, :, np.newaxis]
    if r.shape[2] == 4:
        return result
    if r.shape[:2] != o.shape[:2]:
        return result
    return np.concatenate([to_rgb(r), o[:, :, 3:4]], axis=-1)


def mask_blend(orig, result, mask):
    """Composite result onto orig using mask as alpha. Returns result unchanged if mask is None.

    Raises a shape-mismatch ``ValueError`` naming both arrays when the mask does
    not cover the result. Without this the numpy broadcast failure surfaces as
    ``operands could not be broadcast together with shapes (96,96,1) (50,50,3)``
    — which names neither the node nor which side is wrong. The engine rejects a
    mask whose size differs from the *incoming* image before ``execute()`` runs
    (see ``_mask_shape_error`` in core/pipeline.py); this guard catches the
    remaining case, where the node itself changed the image's size and then
    blended against the original mask.
    """
    if mask is None:
        # No mask still has to reconcile channels. The house pattern for a
        # colour node is `orig_image = image; image = to_rgb(image); ...;
        # mask_blend(orig_image, result, mask)`, which relies on the
        # alpha-restoring branch below — and returning `result` here skipped
        # it, so every such node silently dropped the alpha channel whenever
        # no mask happened to be connected. The masked path restored alpha
        # and the unmasked path did not, which is why it survived: the
        # obvious test (feed RGBA, check the output) passes with a mask
        # wired.
        return restore_alpha(orig, result)
    if (getattr(result, "ndim", 0) >= 2 and getattr(mask, "ndim", 0) >= 2
            and result.shape[:2] != mask.shape[:2]):
        rh, rw = result.shape[:2]
        mh, mw = mask.shape[:2]
        raise ValueError(
            f"mask_blend shape mismatch: result is {rw}x{rh} but mask is {mw}x{mh}. "
            f"The node changed the image size without resizing or dropping the mask — "
            f"return the resized mask, or return mask=None, instead of passing the "
            f"original one through."
        )
    # Channel-count reconciliation.
    #
    # The SPATIAL mismatch above stays a hard error because it is genuinely
    # ambiguous — you cannot tell from here which upstream branch diverged, and
    # silently resizing would hide the actual bug. A CHANNEL mismatch carries no
    # such ambiguity: one side simply has an alpha channel and the other does
    # not, and there is exactly one sensible composite — blend the colour, keep
    # the alpha.
    #
    # Without this, any node whose maths produces a 3-channel result (most colour
    # nodes) died here on RGBA input with "operands could not be broadcast
    # together with shapes (96,96,4) (96,96,3)". Six did: black_and_white, clahe,
    # clouds, duotone, equalize, highlight_shadow. Fixing it once here also stops
    # the next such node from reintroducing it, and covers grayscale-vs-RGB.
    orig_a = np.asarray(orig)
    res_a = np.asarray(result)
    o_ch = orig_a.shape[2] if orig_a.ndim == 3 else 1
    r_ch = res_a.shape[2] if res_a.ndim == 3 else 1
    alpha = None
    if o_ch != r_ch:
        # Prefer the base's alpha: `orig` is what is being composited onto.
        if o_ch == 4:
            alpha = orig_a[:, :, 3:4]
        elif r_ch == 4:
            alpha = res_a[:, :, 3:4]
        orig_a, res_a = to_rgb(orig_a), to_rgb(res_a)

    # A plain (H, W) grayscale side (no explicit channel axis) has no axis
    # for mask's [..., np.newaxis] to broadcast against: (H,W,1) * (H,W)
    # aligns from the right and silently expands to (H,W,W) instead of
    # (H,W,1) — nodes that skip to_rgb() and stay 2-D hit this. Give both
    # sides an explicit trailing channel axis for the arithmetic, then drop
    # it again if neither side had one to begin with.
    was_2d = orig_a.ndim == 2 and res_a.ndim == 2 and alpha is None
    if orig_a.ndim == 2:
        orig_a = orig_a[:, :, np.newaxis]
    if res_a.ndim == 2:
        res_a = res_a[:, :, np.newaxis]

    mask_f = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., np.newaxis]
    blended = mask_f * res_a.astype(np.float32) + (1.0 - mask_f) * orig_a.astype(np.float32)
    # Rounded, not truncated. This line is on the path of nearly every node
    # in the set, so a floor here was a systematic downward bias applied once
    # per masked node in a chain. See to_uint8.
    out = to_uint8(blended)
    if was_2d:
        out = out[:, :, 0]
    if alpha is not None:
        out = np.concatenate([out, alpha], axis=-1)
    return out


def safe_gradient(arr, axis: int):
    """``np.gradient`` along one axis, returning zeros instead of raising when
    that axis is too short to differentiate.

    ``np.gradient`` needs at least ``edge_order + 1`` (i.e. 2) samples along the
    axis and otherwise raises *"Shape of array too small to calculate a numerical
    gradient, at least (edge_order + 1) elements are required."*

    A 1-px-wide or 1-px-tall image reaches these nodes for real: `resize` at its
    declared minimum emits 1×1, and that propagates through every downstream node
    in the graph. The rate of change across a single sample is genuinely zero, so
    zero is the correct answer, not an error — the alternative (raising) violates
    the engine's "no combination of nodes may fail" contract for a case the UI
    can reach with two slider drags.

    Use this anywhere `np.gradient` is called on image data. Five nodes shared
    this exact crash when the fuzzer started drawing params at their declared
    bounds (`edge_segment`, `gradient_threshold`, `noise_estimation`,
    `sharpness_score`, `gradient_magnitude`), and two more carried it latently
    (`edge_detect`, `anisotropic_diffusion`).

    Returns a float array, matching ``np.gradient``'s own promotion: float input
    keeps its precision, integer and bool input become float64.
    """
    a = np.asarray(arr)
    if a.ndim == 0 or a.shape[axis] < 2:
        return np.zeros(a.shape, dtype=a.dtype if a.dtype.kind == "f" else np.float64)
    return np.gradient(a, axis=axis)
