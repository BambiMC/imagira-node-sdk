"""
Base node class for the modular node architecture.

Each node type subclasses BaseNode and provides:
  - type: unique identifier string
  - label: display name
  - category: grouping (Detection, Removal, Post-Process, etc.)
  - color: UI accent colour
  - param_schema: list of parameter definitions
  - execute(): the actual processing logic

This module is the plugin contract itself: extracted verbatim from Imagira's
``core/nodes/base_node.py`` so that third-party node packages depend on a
small, semver'd surface instead of the whole application. The host app
re-exports these same objects, so there is exactly one ``BaseNode`` class and
one ``NODE_REGISTRY`` per process regardless of which import path a node used.
See ``docs/versioning.md`` for what is frozen and what may still grow.
"""

import numpy as np
from enum import Enum

from .expression import ExprParams


# Global node registry: {node_type: node_class}
NODE_REGISTRY: dict[str, type] = {}


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
    produces_data    = False  # green data output port

    # Normally the image port is auto-marked required (the host app inserts
    # "image" into required_inputs/required_outputs when it enumerates node
    # types) whenever the node has that port and isn't a source/sink. Set True
    # on nodes where the image connection itself should stay optional — e.g. a
    # routing node that should also work as a pure mask/samples/data
    # pass-through with no image wired in at all.
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
    # i18n — optional per-node translation overrides (host-app UI feature; see docs/authoring.md).
    # Structure: {"de": {"label": "…", "description": "…", "params": {"key": {"label": "…", "help": "…"}}}}
    # English class attributes are the canonical source; this dict only contains overrides.
    # Missing keys fall back silently to the English value. Option *values* are never translated.
    # IMPORTANT: subclasses that provide translations must define their own `i18n = {...}`.
    # Never mutate this dict at runtime — it is a shared class-level object.
    i18n: dict = {}

    # Topology flags — read by the host app (backend and editor) so it never
    # has to hardcode type strings
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

        # Validate required_inputs / required_outputs consistency at class
        # definition time so misconfigured nodes are caught at import, not at
        # runtime.  Only meaningful on concrete subclasses that declare a type.
        if not getattr(cls, "type", "base") or cls.type == "base":
            return
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

    def __init__(self, node_id, params=None):
        self.node_id = node_id
        # ExprParams (not a plain dict) transparently resolves n8n-style
        # `{{ $json.path }}` templates on read against whatever data the
        # current execute() call is bound to — see expression.py.
        self.params = ExprParams(self._DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.visible = True
        self.compute_backend = ComputeBackend.CPU
        # Per-node free-text annotation ("why this node, why these settings").
        # Persisted in the workflow JSON so it travels with the graph.
        self.note: str = ""

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
        node's own responsibility. The SDK ships no GPU transfer helpers — a node
        that needs them either brings its own or stays CPU-only.
        """
        return self.execute(image, mask=mask, **kwargs)

    def get_param_value(self, key):
        return self.params.get(key)

    def set_param_value(self, key, value):
        self.params[key] = value

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
        note = data.get("note")
        if isinstance(note, str):
            node.note = note
        return node

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.node_id}>"


def register(cls):
    """Register a node class in the global NODE_REGISTRY."""
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
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.concatenate([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image


def as_bool_mask(mask):
    """Return a strict boolean view of *mask*, or None.

    Masks legitimately arrive as ``bool``, ``uint8`` (0/1 or 0/255), or a float
    soft matte in [0, 1] — see "The mask stream" in docs/data_streams.md. Only
    ``bool`` works as a *selection*: ``image[mask]`` with a uint8 mask is
    integer indexing, so it silently returns the wrong pixels or dies with
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


def mask_blend(orig, result, mask):
    """Composite result onto orig using mask as alpha. Returns result unchanged if mask is None.

    Raises a shape-mismatch ``ValueError`` naming both arrays when the mask does
    not cover the result. Without this the numpy broadcast failure surfaces as
    ``operands could not be broadcast together with shapes (96,96,1) (50,50,3)``
    — which names neither the node nor which side is wrong. A host app is
    expected to reject a mask whose size differs from the *incoming* image
    before ``execute()`` runs; this guard catches the remaining case, where the
    node itself changed the image's size and then blended against the original
    mask.
    """
    if mask is None:
        return result
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
    mask_f = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., np.newaxis]
    blended = mask_f * result.astype(np.float32) + (1.0 - mask_f) * orig.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)
