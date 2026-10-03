"""Node kinds -- base classes that own the boilerplate of the common node shapes.

A kind fixes the port flags, the ``execute`` wrapper (null guard, input
normalisation, mask handling, output cast) and the default port docs of one node
shape; the author writes only the part that differs::

    class Brightness(Filter):
        type = "brightness"
        brightness = Slider(0, -150, 150, step=1, help="...")

        def apply(self, rgb, p):
            return rgb + p.brightness

Kinds: ``Filter`` ``Geometry`` ``MaskMaker`` ``MaskOp`` ``Sampler`` ``Analyzer``
``DataOp`` ``Merge`` ``SampleConsumer`` (a wrapped ``execute``), and ``Node``
``Source`` ``Sink`` (flag presets: the author still writes ``execute``). See
analysis/54_quality.md 5.14 for the contract; this docstring does not repeat it.

Mechanics
---------
* A kind never registers itself: it keeps the inherited ``type = "base"``.
* A kind hooks in through ``__init_subclass__`` and runs BEFORE
  ``BaseNode.__init_subclass__`` (``super()`` is called after the schema and the
  flags are in place), so BaseNode's validation sees the finished class.
* A flag the kind provides is set only when no class body between the subclass
  and the kind sets it: explicit wins.
* The author's method signature is inspected once per class; the kind passes
  only the optional arguments (``data``, ``mask``, ``is_mask``, ...) it names.
* ``p`` is a read-only :class:`ParamView`; ``fixed_params`` overlays pinned values.
* ``null_result`` adds keys to whatever a kind returns from a guard path.
"""

from __future__ import annotations

import inspect

import numpy as np

try:
    from core.nodes.base_node import (BaseNode, as_bool_mask, mask_blend, mask_from_uint8,
                                      mask_is_soft, mask_to_uint8, restrict_mask_selection,
                                      restore_alpha as _restore_alpha, to_rgb, to_rgb_keep_alpha,
                                      to_uint8)
    from core.nodes.params import ParamView, params_from_class
except ImportError:   # vendored copy (node SDK)
    from .base_node import (BaseNode, as_bool_mask, mask_blend, mask_from_uint8,
                            mask_is_soft, mask_to_uint8, restrict_mask_selection,
                            restore_alpha as _restore_alpha, to_rgb, to_rgb_keep_alpha,
                            to_uint8)
    from .params import ParamView, params_from_class

__all__ = [
    "Node", "Filter", "Geometry", "MaskMaker", "MaskOp", "Sampler", "Analyzer",
    "DataOp", "Merge", "SampleConsumer", "Source", "Sink", "masked_pixels",
]

# -- standard port docs ------------------------------------------------------
# Copied verbatim from the most common text in the node set (analysis/54 5.14),
# so a migrated node's metadata stays identical when it takes the default.
_IN_IMAGE = ("The image to process. Its pixels are what this node's effect is applied to; "
             "the result leaves through the Image output.")
_IN_MASK_LIMITER = ("Optional region limiter. The effect is computed on the whole image and then "
                    "composited back through the mask (`mask_blend`), so pixels outside the mask "
                    "keep their original values and a soft (float) mask fades the effect "
                    "proportionally.")
_IN_MASK_SELECTION = ("Optional region limiter. The node reads the mask as a pixel selection "
                      "(`as_bool_mask`) and restricts what it looks at / changes to those pixels.")
_IN_MASK_SUBJECT = ("Required. The mask this node works on — here the mask is the subject, not a "
                    "region limiter.")
_IN_DATA = ("Optional. Structured data from an upstream Data output. `execute()` does not read it; "
            "it is bound to this node's parameters so any parameter can be driven from upstream "
            "with a `{{ $json.key }}` expression (core/expression.py). This is the default meaning "
            "of the green input across the node set.")
_IN_DATA_WHOLE = ("Optional. `execute()` reads the incoming value as a whole, without looking up a "
                  "fixed key. As on every node, the same value is also bound to `{{ $json.key }}` "
                  "parameter expressions.")
_OUT_IMAGE = "The processed image — this node's primary product."
_OUT_MASK_PRODUCT = "The mask this node computes — its primary product."
_OUT_DATA_MEASURED = ("The measured values, as a data object — the same numbers the node reports "
                      "internally, but on a wire other nodes can actually read. Bind them into any "
                      "downstream parameter with a `{{ $json.key }}` expression.")
_IN_SAMPLES_PASS = ("The sample set to work from (`{\"islands\": [...]}`, see NODE_RULES.md § Sample "
                    "stream). The stream is forwarded on the Samples output, so later nodes can use "
                    "the same samples.")
_IN_SAMPLES_TERMINAL = ("The sample set to work from (`{\"islands\": [...]}`, see NODE_RULES.md § "
                        "Sample stream). This node consumes the stream: because it accepts samples "
                        "without producing them, the pipeline clears the stream afterwards, so "
                        "downstream nodes get no stale samples.")
_OUT_SAMPLES_PASS = ("The sample set passed on after this node used it, so further nodes can read "
                     "the same samples.")

# docs key -> the flag that must be on for the port to exist (BaseNode's own table)
_PORT_FLAG = {
    "in.image": "has_input_port", "in.mask": "accepts_mask", "in.samples": "accepts_samples",
    "in.data": "accepts_data", "out.image": "has_output_port", "out.mask": "produces_mask",
    "out.samples": "produces_samples", "out.data": "produces_data",
}


def _gpu_ready() -> bool:
    try:
        from core.gpu import cupy_available
    except ImportError:
        return False
    return bool(cupy_available())


def _cuda_ready() -> bool:
    try:
        from core.gpu import cv2_cuda_available
    except ImportError:
        return False
    return bool(cv2_cuda_available())


def _host(x):
    """A CuPy array (or anything array-like) as a NumPy array; NumPy passes through."""
    if x is None or isinstance(x, np.ndarray):
        return x
    import cupy as cp
    return cp.asnumpy(x)


def _split(result, primary):
    """``result`` is the primary product, or a dict of outputs that carries it under *primary*."""
    if isinstance(result, dict):
        extras = dict(result)
        return extras.pop(primary, None), extras
    return result, {}


def masked_pixels(rgb, mask):
    """``(pixels, positions)`` of the selected pixels of an RGB image.

    Selected = ``np.where(mask)`` when the mask has any pixel set, otherwise the
    whole frame (an empty or absent mask means "no restriction"). ``pixels`` is
    ``(N, 3)``, ``positions`` ``(N, 2)`` as ``[y, x]``. The semantics every
    sampler used to repeat (dominant_color).
    """
    h, w = rgb.shape[:2]
    if mask is not None and np.any(mask):
        ys, xs = np.where(mask)
        pixels = rgb[ys, xs, :3]
        positions = np.stack([ys, xs], axis=1)
    else:
        all_y, all_x = np.mgrid[0:h, 0:w]
        pixels = rgb.reshape(-1, 3)
        positions = np.stack([all_y.ravel(), all_x.ravel()], axis=1)
    return pixels, positions


# -- machinery ---------------------------------------------------------------

def _kind_of(cls):
    for k in cls.__mro__:
        if "_kind_root" in k.__dict__ and k is not _Kind:
            return k
    return None


def _explicit(cls, kind, flag) -> bool:
    """Does a class body between *cls* and *kind* set *flag* itself (not derived by a kind)?"""
    for k in cls.__mro__:
        if k is kind:
            return False
        if flag in k.__dict__ and flag not in k.__dict__.get("_kind_set", ()):
            return True
    return False


class _Kind(BaseNode):
    """Common base of the kinds. Not a kind itself."""

    _kind_root = True
    # (method name, number of fixed positional params, optional names allowed) or None
    _author: tuple | None = None
    _gpu_capable = False
    _uses_mask_toggle = False
    # What the author's method names, filled per class by _prepare.
    _author_extras: frozenset = frozenset()
    _author_name: str = ""
    # Name of an optional host-array GPU variant (Filter: ``apply_cuda``) or None.
    _cuda_name: str | None = None
    # Extra keys merged into the dict a kind returns from any of its guard paths
    # (image/mask/data missing, empty selection, ``requires_mask``): the keys a
    # hand-written node's null result carried (``viz``, ``data``, ...).
    null_result: dict = {}
    # Values ``ParamView`` serves instead of the node's own params: a hidden or
    # legacy subclass pins settings without overriding ``execute`` (A6). A
    # callable value is called with the un-overlaid view and returns the value.
    fixed_params: dict = {}

    @classmethod
    def _flag_defaults(cls) -> dict:
        return {}

    @classmethod
    def _default_docs(cls) -> dict:
        return {}

    def __init_subclass__(cls, **kwargs):
        if "_kind_root" in cls.__dict__:          # a kind being defined
            super().__init_subclass__(**kwargs)
            return
        kind = _kind_of(cls)
        if kind is not None:
            _prepare(cls, kind)
        super().__init_subclass__(**kwargs)
        if kind is not None:
            _finish(cls)

    # -- helpers for the kind wrappers ---------------------------------------
    def _author_fn(self, gpu=False):
        if gpu == "cuda":
            return getattr(self, self._cuda_name)
        return getattr(self, self._author_name + ("_gpu" if gpu else ""))

    def _opt(self, **candidates):
        names = self._author_extras
        return {k: v for k, v in candidates.items() if k in names}

    def _has_gpu_variant(self) -> bool:
        return self._gpu_capable and self.supports_gpu and hasattr(self, self._author_name + "_gpu")

    def _has_cuda_variant(self) -> bool:
        return bool(self._cuda_name and self.supports_gpu and hasattr(self, self._cuda_name))

    def _null(self, out: dict) -> dict:
        """*out* (a guard-path result) plus the class's ``null_result`` keys."""
        if self.null_result:
            out.update(self.null_result)
        return out


def _inherited_own_docs(cls, kind) -> dict:
    for k in cls.__mro__[1:]:
        if k is kind:
            break
        if "_own_port_docs" in k.__dict__:
            return dict(k.__dict__["_own_port_docs"])
        if "port_docs" in k.__dict__:
            return dict(k.__dict__["port_docs"])
    return {}


def _prepare(cls, kind) -> None:
    author = kind._author
    if author is not None:
        for forbidden in ("execute", "execute_gpu", "execute_multiple"):
            if forbidden in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__}: a {kind.__name__} node does not define {forbidden}() -- "
                    f"the kind provides it; write {author[0]}() instead")

    # A subclass of a finished node inherits that node's MERGED port docs, whose
    # defaults may name ports this subclass switches off; start again from the
    # docs the parent's author wrote.
    if "port_docs" not in cls.__dict__:
        cls.port_docs = _inherited_own_docs(cls, kind)

    schema = params_from_class(cls)
    if schema is not None:
        cls.param_schema = schema
        cls._params_derived = True

    # flags: only what the author did not set
    defaults = dict(cls._flag_defaults())
    if kind._uses_mask_toggle and getattr(cls, "uses_mask", True) is False:
        defaults["accepts_mask"] = False
    derived = []
    for flag, value in defaults.items():
        if not _explicit(cls, kind, flag):
            setattr(cls, flag, value)
            derived.append(flag)

    # description: the class docstring, unless a class body sets one
    doc = cls.__dict__.get("__doc__")
    if doc and not _explicit(cls, kind, "description"):
        cls.description = inspect.cleandoc(doc)
        derived.append("description")

    if author is not None:
        _inspect_author(cls, kind, author)
        gpu_name = author[0] + "_gpu"
        has_gpu = hasattr(cls, gpu_name) or bool(kind._cuda_name and hasattr(cls, kind._cuda_name))
        if kind._gpu_capable and has_gpu and not _explicit(cls, kind, "supports_gpu"):
            cls.supports_gpu = True
            derived.append("supports_gpu")
    cls._kind_set = tuple(derived)
    if (author is not None and "image" in cls._author_extras
            and getattr(cls, "type", "base") != "base"
            and ("has_input_port" in derived or not cls.has_input_port)):
        raise TypeError(f"{cls.__name__}: {author[0]}() names `image`, so the class must "
                        f"declare has_input_port = True")


def _inspect_author(cls, kind, author) -> None:
    name, n_fixed, allowed = author
    fn = getattr(cls, name, None)
    if fn is None:
        if getattr(cls, "type", "base") != "base":
            raise TypeError(f"{cls.__name__}: a {kind.__name__} node must define {name}()")
        cls._author_name = name
        cls._author_extras = frozenset()
        return
    cls._author_name = name
    cls._author_extras = _extras_of(cls, name, fn, n_fixed, allowed)
    gpu_fn = getattr(cls, name + "_gpu", None)
    if gpu_fn is not None and kind._gpu_capable:
        if _extras_of(cls, name + "_gpu", gpu_fn, n_fixed, allowed) != cls._author_extras:
            raise TypeError(f"{cls.__name__}: {name}_gpu() must take the same optional arguments "
                            f"as {name}()")
    cuda_fn = getattr(cls, kind._cuda_name, None) if kind._cuda_name else None
    if cuda_fn is not None:
        if _extras_of(cls, kind._cuda_name, cuda_fn, n_fixed, allowed) != cls._author_extras:
            raise TypeError(f"{cls.__name__}: {kind._cuda_name}() must take the same optional "
                            f"arguments as {name}()")
    if "mask" in cls._author_extras and not getattr(cls, "accepts_mask", False):
        raise TypeError(f"{cls.__name__}: {name}() names `mask` but accepts_mask is False")
    if "samples" in cls._author_extras and not getattr(cls, "accepts_samples", False):
        raise TypeError(f"{cls.__name__}: {name}() names `samples` but accepts_samples is False")


def _extras_of(cls, name, fn, n_fixed, allowed) -> frozenset:
    params = list(inspect.signature(fn).parameters.values())[1:]      # drop self
    for prm in params:
        if prm.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"{cls.__name__}.{name}(): *args/**kwargs are not allowed")
    if len(params) < n_fixed:
        raise TypeError(f"{cls.__name__}.{name}() needs {n_fixed} positional parameters, "
                        f"has {len(params)}")
    extras = [prm.name for prm in params[n_fixed:]]
    bad = [e for e in extras if e not in allowed]
    if bad:
        raise TypeError(f"{cls.__name__}.{name}(): unexpected parameter(s) {bad}; "
                        f"optional ones allowed here: {sorted(allowed)}")
    return frozenset(extras)


def _finish(cls) -> None:
    """Merge the kind's default port docs under the class's own (explicit wins)."""
    if getattr(cls, "type", "base") == "base":
        return
    docs = {k: v for k, v in cls._default_docs().items()
            if getattr(cls, _PORT_FLAG[k], False)}
    own = dict(cls.__dict__.get("port_docs") or {})
    docs.update(own)
    cls._own_port_docs = own
    cls.port_docs = docs


# ----------------------------------------------------------------------------
# Node / Source / Sink: flag presets, the author writes execute()
# ----------------------------------------------------------------------------

class Node(_Kind):
    """A plain node that can declare params as descriptors. Control flow, loops,
    routing: the author writes ``execute`` and sets every flag."""

    _kind_root = True


class Source(_Kind):
    """A node with no image input that starts a branch (readers, generators)."""

    _kind_root = True
    trigger = False        # True: the node starts a run (``is_trigger``)

    @classmethod
    def _flag_defaults(cls):
        d = {"has_input_port": False, "is_source_node": True}
        if cls.trigger:
            d["is_trigger"] = True
        return d


class Sink(_Kind):
    """A terminal node: no output port (exporters, previews)."""

    _kind_root = True
    writes_files = False   # True: ``is_disk_exporter``

    @classmethod
    def _flag_defaults(cls):
        d = {"is_sink_node": True, "has_output_port": False}
        if cls.writes_files:
            d["is_disk_exporter"] = True
        return d


# ----------------------------------------------------------------------------
# Filter
# ----------------------------------------------------------------------------

class Filter(_Kind):
    """Image in, image out, optionally limited by a mask.

    Author: ``apply(self, rgb, p[, data][, mask])`` -> array (float or uint8), or
    a dict with the array under ``"image"`` plus extra outputs. ``rgb`` is
    float32 RGB unless ``float_input = False`` (then uint8 RGB). If ``apply``
    names ``mask`` the author owns the mask: no ``mask_blend`` (``blend_mask =
    True`` keeps the blend as well; ``restore_alpha = True`` puts the input's
    alpha back instead). ``requires_mask = True`` returns the untouched image
    when there is no (or an empty) mask. Optional ``apply_gpu`` with the same
    signature receives and returns CuPy arrays; optional ``apply_cuda`` gets
    and returns host uint8 arrays (cv2.cuda), tried when CuPy is not used and
    ``cv2_cuda_available()``; any exception falls back to the CPU ``apply``.
    """

    _kind_root = True
    _author = ("apply", 2, frozenset({"data", "mask"}))
    _gpu_capable = True
    _uses_mask_toggle = True
    _cuda_name = "apply_cuda"
    float_input = True
    uses_mask = True
    # Who composites the effect through the mask. None: the kind does (`mask_blend`)
    # unless `apply` names `mask`, then the author owns it. True: `apply` gets the
    # mask AND the kind still blends (statistics from the masked region, effect
    # composited). False: the kind never blends.
    blend_mask: bool | None = None
    # True: after `to_uint8` the kind puts the input's alpha back (`restore_alpha`)
    # when the author owns the mask and so no `mask_blend` does it.
    restore_alpha: bool = False
    # True: without a mask, or with an empty one, the image goes back untouched
    # (the very same object) and `apply` is not called.
    requires_mask: bool = False

    @classmethod
    def _flag_defaults(cls):
        return {"accepts_mask": True, "produces_image": True}

    @classmethod
    def _default_docs(cls):
        return {
            "in.image": _IN_IMAGE, "in.mask": _IN_MASK_LIMITER, "in.data": _IN_DATA,
            "out.image": _OUT_IMAGE}

    def execute(self, image, mask=None, data=None, context=None):
        return self._run(image, mask, data, False)

    def execute_gpu(self, image, mask=None, data=None, context=None):
        # Order: CuPy variant (`apply_gpu`), then the host-array cv2.cuda variant
        # (`apply_cuda`, any failure falls back), then the CPU path.
        if _gpu_ready() and self._has_gpu_variant():
            return self._run(image, mask, data, True)
        if self._has_cuda_variant() and _cuda_ready():
            try:
                return self._run(image, mask, data, "cuda")
            except Exception:
                pass
        return self.execute(image, mask=mask, data=data, context=context)

    def _run(self, image, mask, data, gpu):
        if image is None:
            return self._null({"image": None, "mask": mask})
        mask_in = mask if self.accepts_mask else None
        if self.requires_mask and (mask_in is None or not np.any(mask_in)):
            return self._null({"image": image, "mask": mask})
        rgb = to_rgb(image)
        if gpu == "cuda":
            arg = rgb
        elif gpu:
            import cupy as cp
            arg = cp.asarray(rgb, dtype=cp.float32) if self.float_input else cp.asarray(rgb)
        else:
            arg = rgb.astype(np.float32) if self.float_input else rgb
        res = self._author_fn(gpu)(arg, ParamView(self), **self._opt(data=data, mask=mask_in))
        res, extras = _split(res, "image")
        if gpu is True and not isinstance(res, np.ndarray):
            import cupy as cp
            if res.dtype != cp.uint8:
                res = cp.rint(cp.clip(res, 0, 255)).astype(cp.uint8)
            res = cp.asnumpy(res)
        res = to_uint8(res)
        blend = self.blend_mask if self.blend_mask is not None else "mask" not in self._author_extras
        if blend:
            res = mask_blend(image, res, mask_in)
        elif self.restore_alpha:
            res = _restore_alpha(image, res)
        out = {"image": res, "mask": mask}
        out.update(extras)
        return out


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

class Geometry(_Kind):
    """Moves pixels: the image and an incoming mask go through the same transform.

    Author: ``transform(self, arr, p[, is_mask][, soft][, data])`` -> array. It is
    called once for the image (uint8 RGB, or RGBA with its alpha kept) and, when
    a mask arrives, once for the mask as a uint8 0-255 picture with
    ``is_mask=True`` (``soft`` says whether the original mask was a soft matte,
    for choosing a resampler). The kind converts the mask back with
    ``mask_from_uint8`` and pads/crops it to the image's size when the transform
    rounds the two differently. Optional ``transform_gpu`` (CuPy in/out).
    ``mask_as_uint8 = False`` hands the mask over in its own dtype instead (and
    takes it back as returned): use it for lossless index maps such as a flip.
    """

    _kind_root = True
    _author = ("transform", 2, frozenset({"is_mask", "soft", "data"}))
    _gpu_capable = True
    _uses_mask_toggle = True
    uses_mask = True
    # True: the mask goes through ``transform`` as a uint8 0-255 picture (a
    # resampling transform needs that). False: the mask goes through in its own
    # dtype and comes back unconverted -- for lossless index maps (flip, shift,
    # transpose) where an 8-bit round trip would quantise a soft float matte.
    mask_as_uint8 = True

    @classmethod
    def _flag_defaults(cls):
        return {
            "accepts_mask": True, "produces_mask": True, "produces_image": True}

    @classmethod
    def _default_docs(cls):
        return {
            "in.image": _IN_IMAGE,
            "in.mask": ("Optional. A mask wired in goes through the same change as the image, so mask "
                        "and image stay aligned for the nodes downstream."),
            "in.data": _IN_DATA, "out.image": _OUT_IMAGE,
            "out.mask": ("The mask, transformed by the same change as the image, so the two stay "
                         "aligned for the nodes downstream. Absent means no mask was wired in.")}

    def execute(self, image, mask=None, data=None, context=None):
        return self._run(image, mask, data, False)

    def execute_gpu(self, image, mask=None, data=None, context=None):
        if not (_gpu_ready() and self._has_gpu_variant()):
            return self.execute(image, mask=mask, data=data, context=context)
        return self._run(image, mask, data, True)

    def _xf(self, arr, p, gpu, **opt):
        if gpu:
            import cupy as cp
            arr = cp.asarray(arr)
        return self._author_fn(gpu)(arr, p, **self._opt(**opt))

    def _run(self, image, mask, data, gpu):
        if image is None:
            return self._null({"image": None, "mask": mask})
        p = ParamView(self)
        res = self._xf(to_rgb_keep_alpha(image), p, gpu, is_mask=False, soft=False, data=data)
        res, extras = _split(res, "image")
        res = to_uint8(_host(res))
        out_mask = None
        if mask is not None and self.accepts_mask:
            soft = mask_is_soft(mask)
            marg = mask_to_uint8(mask) if self.mask_as_uint8 else np.asarray(mask)
            mres = self._xf(marg, p, gpu, is_mask=True, soft=soft, data=data)
            if isinstance(mres, dict):
                mres = mres.get("image")
            out_mask = np.asarray(_host(mres))
            if self.mask_as_uint8:
                out_mask = mask_from_uint8(out_mask, soft)
            if out_mask.shape != res.shape[:2]:
                out_mask = np.pad(out_mask, ((0, max(0, res.shape[0] - out_mask.shape[0])),
                                             (0, max(0, res.shape[1] - out_mask.shape[1]))),
                                  mode="constant", constant_values=False)
                out_mask = out_mask[:res.shape[0], :res.shape[1]]
        out = {"image": res, "mask": out_mask}
        out.update(extras)
        return out


# ----------------------------------------------------------------------------
# MaskMaker / MaskOp
# ----------------------------------------------------------------------------

class MaskMaker(_Kind):
    """Image in, mask out (the image port is not passed on).

    Author: ``make(self, rgb, p[, data][, mask][, samples])`` -> mask (or a dict with it
    under ``"mask"``). ``rgb`` is uint8 RGB (float32 with ``float_input =
    True``). The result is restricted to the incoming mask with
    ``restrict_mask_selection`` unless ``make`` names ``mask`` (then the author
    decides). Optional ``make_gpu`` (CuPy in/out).
    """

    _kind_root = True
    _author = ("make", 2, frozenset({"data", "mask", "samples"}))
    _gpu_capable = True
    _uses_mask_toggle = True
    float_input = False
    uses_mask = True

    @classmethod
    def _flag_defaults(cls):
        return {
            "accepts_mask": True, "produces_mask": True, "produces_image": False,
            "has_output_port": False, "required_outputs": ("mask",)}

    @classmethod
    def _default_docs(cls):
        return {
            "in.image": ("The image the selection is computed from. It is not modified — the node's "
                         "product is the Mask output."),
            "in.mask": ("Optional region limiter. Only pixels inside the mask can end up in this "
                        "node's own selection (`restrict_mask_selection`); everything outside it is "
                        "cleared, whatever the node's own thresholding decided."),
            "in.data": _IN_DATA, "out.mask": _OUT_MASK_PRODUCT}

    def execute(self, image, mask=None, data=None, context=None, samples=None):
        return self._run(image, mask, data, False, samples)

    def execute_gpu(self, image, mask=None, data=None, context=None, samples=None):
        if not (_gpu_ready() and self._has_gpu_variant()):
            return self.execute(image, mask=mask, data=data, context=context, samples=samples)
        return self._run(image, mask, data, True, samples)

    def _run(self, image, mask, data, gpu, samples=None):
        mask_in = mask if self.accepts_mask else None
        if image is None:
            # A node without a mask port has no mask to relay (NODE_RULES 3).
            return self._null({"image": None, "mask": mask_in})
        rgb = to_rgb(image)
        if gpu:
            import cupy as cp
            arg = cp.asarray(rgb, dtype=cp.float32) if self.float_input else cp.asarray(rgb)
        else:
            arg = rgb.astype(np.float32) if self.float_input else rgb
        res = self._author_fn(gpu)(arg, ParamView(self),
                                   **self._opt(data=data, mask=mask_in, samples=samples))
        res, extras = _split(res, "mask")
        res = _host(res)
        if "mask" not in self._author_extras:
            res = restrict_mask_selection(res, mask_in, self.type)
        out = {"image": None, "mask": res}
        out.update(extras)
        return out


class MaskOp(_Kind):
    """Mask in, mask out; no image ports.

    Author: ``apply(self, mask, p[, data][, image])`` -> mask (or a dict with it
    under ``"mask"``). A missing mask, or an empty one unless ``call_on_empty =
    True``, returns ``{"image": None, "mask": None}`` without calling ``apply``.
    Naming ``image`` hands over the raw image (the class must declare
    ``has_input_port = True``); a missing image then returns ``{"image": None,
    "mask": mask}``. Optional ``apply_gpu`` (CuPy in/out).
    """

    _kind_root = True
    _author = ("apply", 2, frozenset({"data", "image"}))
    _gpu_capable = True
    call_on_empty = False

    @classmethod
    def _flag_defaults(cls):
        return {
            "has_input_port": False, "has_output_port": False, "produces_image": False,
            "accepts_mask": True, "produces_mask": True,
            "required_inputs": ("mask",), "required_outputs": ("mask",)}

    @classmethod
    def _default_docs(cls):
        return {
            "in.mask": _IN_MASK_SUBJECT, "in.data": _IN_DATA, "out.mask": _OUT_MASK_PRODUCT}

    def execute(self, image=None, mask=None, data=None, context=None):
        return self._run(image, mask, data, False)

    def execute_gpu(self, image=None, mask=None, data=None, context=None):
        if not (_gpu_ready() and self._has_gpu_variant()):
            return self.execute(image, mask=mask, data=data, context=context)
        return self._run(image, mask, data, True)

    def _run(self, image, mask, data, gpu):
        if mask is None or (not self.call_on_empty and not np.any(mask)):
            return self._null({"image": None, "mask": None})
        if image is None and "image" in self._author_extras:
            return self._null({"image": None, "mask": mask})
        arg = mask
        if gpu:
            import cupy as cp
            arg = cp.asarray(mask)
        res = self._author_fn(gpu)(arg, ParamView(self), **self._opt(data=data, image=image))
        res, extras = _split(res, "mask")
        out = {"image": None, "mask": _host(res)}
        out.update(extras)
        return out


# ----------------------------------------------------------------------------
# Sampler / Analyzer
# ----------------------------------------------------------------------------

class Sampler(_Kind):
    """Image (and optional mask) in, sample islands out.

    Author: ``sample(self, rgb, mask, p[, data])`` -> list of island dicts (or a
    dict with them under ``"islands"``). Use :func:`masked_pixels` for the usual
    "selected pixels or the whole frame". The kind fills ``label`` (index + 1),
    ``area`` (``len(positions)``) and ``mean_color`` only when an island lacks
    them. ``max_pixels`` (class attribute, default None) subsamples each island's
    pixels/positions with ``default_rng(42)`` after ``area`` is recorded;
    ``mean_color`` then describes the kept pixels. ``sample`` returning a bare
    ``None`` means "no sample set": the result carries ``"samples": None``.
    ``passes_mask = True`` relays the incoming mask on the ``mask`` key (guard
    paths included) instead of None.
    """

    _kind_root = True
    _author = ("sample", 3, frozenset({"data"}))
    _uses_mask_toggle = True
    uses_mask = True
    max_pixels: int | None = None
    passes_mask: bool = False

    @classmethod
    def _flag_defaults(cls):
        return {
            "accepts_mask": True, "produces_samples": True, "produces_image": False,
            "has_output_port": False, "required_outputs": ("samples",)}

    @classmethod
    def _default_docs(cls):
        return {
            "in.image": ("The image the sample colours are read from. The samples leave through the "
                         "Samples output."),
            "in.mask": _IN_MASK_SELECTION, "in.data": _IN_DATA,
            "out.samples": ("The sample set this node produces (`{\"islands\": [...]}`), for a Sample "
                            "consumer downstream.")}

    def execute(self, image=None, mask=None, data=None, context=None):
        relay = mask if self.passes_mask else None
        if image is None:
            return self._null({"image": None, "mask": relay,
                               "samples": {"islands": [], "image_size": (0, 0)}})
        rgb = to_rgb(image)
        mask_in = mask if self.accepts_mask else None
        res = self._author_fn()(rgb, mask_in, ParamView(self), **self._opt(data=data))
        if res is None:
            return self._null({"image": None, "mask": relay, "samples": None})
        islands, extras = _split(res, "islands")
        islands = list(islands or [])
        self._complete(islands)
        out = {"image": None, "mask": relay,
               "samples": {"islands": islands, "image_size": rgb.shape[:2]}}
        out.update(extras)
        return out

    def _complete(self, islands):
        limit = self.max_pixels
        rng = np.random.default_rng(42) if limit else None
        for i, isl in enumerate(islands):
            isl.setdefault("label", i + 1)
            pos, px = isl.get("positions"), isl.get("pixels")
            if "area" not in isl and (pos is not None or px is not None):
                isl["area"] = int(len(pos if pos is not None else px))
            if limit and pos is not None and len(pos) > limit:
                idx = rng.choice(len(pos), limit, replace=False)
                isl["positions"] = pos[idx]
                if px is not None:
                    isl["pixels"] = px[idx]
                    px = isl["pixels"]
            if "mean_color" not in isl and px is not None and len(px):
                isl["mean_color"] = to_uint8(np.asarray(px).mean(axis=0)).tolist()


class Analyzer(_Kind):
    """Image (and optional mask) in, measured values out on the data port.

    Author: ``measure(self, rgb, pixels, p[, mask][, data])`` -> dict or None.
    ``pixels`` is the selected ``(N, 3)`` (the whole frame without a mask or with
    an empty one). An empty selection yields ``data: None`` without calling
    ``measure`` -- unless ``call_on_empty = True`` (a node that reports a value
    for an empty frame or selection, e.g. a coverage of 0).
    """

    _kind_root = True
    _author = ("measure", 3, frozenset({"mask", "data"}))
    _uses_mask_toggle = True
    uses_mask = True
    call_on_empty = False

    @classmethod
    def _flag_defaults(cls):
        return {
            "accepts_mask": True, "produces_data": True, "produces_image": False,
            "has_output_port": False}

    @classmethod
    def _default_docs(cls):
        return {
            "in.image": _IN_IMAGE, "in.mask": _IN_MASK_SELECTION, "in.data": _IN_DATA,
            "out.data": _OUT_DATA_MEASURED}

    def execute(self, image, mask=None, data=None, context=None):
        if image is None:
            return self._null({"image": None, "mask": mask, "data": None})
        mask_in = mask if self.accepts_mask else None
        rgb = to_rgb(image)
        if mask_in is not None and np.any(mask_in):
            pixels = rgb[as_bool_mask(mask_in), :3]
        else:
            pixels = rgb[:, :, :3].reshape(-1, 3)
        if len(pixels) == 0 and not self.call_on_empty:
            return self._null({"image": None, "mask": mask, "data": None})
        res = self._author_fn()(rgb, pixels, ParamView(self),
                                **self._opt(mask=mask_in, data=data))
        return {"image": None, "mask": mask, "data": res}


# ----------------------------------------------------------------------------
# DataOp / Merge / SampleConsumer
# ----------------------------------------------------------------------------

class DataOp(_Kind):
    """Data in, data out; no image, mask or sample ports.

    Author: ``apply(self, data, p)`` -> the new data value (a dict, list, text, ...).
    ``data is None`` returns ``{"data": None}`` without calling ``apply``, unless
    ``call_without_data = True``. The result carries only the ``data`` key, like
    the hand-written data nodes (the engine reads a missing image/mask as None).
    """

    _kind_root = True
    _author = ("apply", 2, frozenset())
    call_without_data = False

    @classmethod
    def _flag_defaults(cls):
        return {
            "has_input_port": False, "has_output_port": False, "produces_image": False,
            "accepts_data": True, "produces_data": True,
            "required_inputs": ("data",), "required_outputs": ("data",)}

    @classmethod
    def _default_docs(cls):
        return {"in.data": _IN_DATA_WHOLE}

    def execute(self, image=None, mask=None, data=None, context=None):
        if data is None and not self.call_without_data:
            return self._null({"data": None})
        return {"data": self._author_fn()(data, ParamView(self))}


class Merge(_Kind):
    """Several image edges on one port, combined into one.

    Author: ``combine(self, layers, p[, data][, datas])``; ``layers`` is a list
    of ``(image, mask)`` pairs in connection order, None images dropped AFTER the
    pairing (so a mask keeps its image's slot). Returns an array (the image; the
    first layer's mask is passed on) or a dict of outputs. Fewer than
    ``multi_input_count`` layers pass the first layer through unchanged, like
    the single-edge ``execute``. ``data`` is the first non-None predecessor data,
    ``datas`` the per-predecessor list.

    Flag defaults are the majority of the 23 ``accepts_multiple`` nodes measured
    on 2026-10-02: accepts_mask 16/23 (12/13 of the image-layer nodes),
    produces_mask False 17/23, produces_image True 17/23, has_input_port True
    13/23 (the 10 Falses are the mask and sample mergers, which set it
    explicitly), has_output_port True 12/23 (default). The mask-merging and
    sample-merging nodes override.

    ``reconcile = True`` (opt-in) is the boilerplate the two-layer compositing
    nodes (blend, dissolve, ...) repeated: every layer loses its alpha channel,
    layers after the first are resized to the first one's size (resized WITH
    their alpha, then stripped, as PIL premultiplies), ``combine`` returns the
    new picture (float or uint8 array), and the kind casts it with ``to_uint8``
    and composites it back onto the first layer through the first layer's mask
    (``mask_blend``, which also restores the first layer's alpha). The first
    layer's mask is passed on. A dict result is returned untouched.

    ``stream`` picks what is merged. ``"image"`` (default) is everything above.
    ``"mask"``: ``combine(masks, p[, data][, datas])`` gets the list of the
    predecessors' masks (None entries dropped) and returns the mask (or a dict
    with it under ``"mask"``); no mask at all returns ``{"image": None, "mask":
    None}``; the class flags have no image ports. ``"samples"``:
    ``combine(sample_sets, p[, data][, datas])`` gets the per-predecessor sample
    sets (None entries dropped; possibly an empty list) and returns the samples
    dict (or a dict of outputs with it under ``"samples"``; a samples dict has
    no ``"samples"`` key, so the two are told apart). Neither stream applies
    ``multi_input_count`` or ``reconcile``: ``combine`` decides what "too few"
    means. The single-edge ``execute`` passes the one input through unchanged,
    unless ``single_edge_combines = True`` runs ``combine`` on it as well (a
    union of one set still merges its islands).
    """

    _kind_root = True
    _author = ("combine", 2, frozenset({"data", "datas"}))
    reconcile = False
    stream = "image"        # "image" | "mask" | "samples"
    # Mask / samples streams: True runs ``combine`` on the single edge too (a union
    # of one set still merges its islands); False passes the one input through.
    single_edge_combines = False

    @classmethod
    def _flag_defaults(cls):
        if cls.stream == "mask":
            return {
                "accepts_multiple": True, "accepts_mask": True, "produces_mask": True,
                "produces_image": False, "has_input_port": False, "has_output_port": False,
                "required_inputs": ("mask",), "required_outputs": ("mask",)}
        if cls.stream == "samples":
            return {
                "accepts_multiple": True, "accepts_samples": True, "accepts_mask": False,
                "produces_samples": True, "produces_image": False, "has_input_port": False,
                "has_output_port": False, "required_inputs": ("samples",),
                "required_outputs": ("samples",)}
        return {
            "accepts_multiple": True, "accepts_mask": True, "produces_image": True,
            "has_input_port": True}

    @classmethod
    def _default_docs(cls):
        if cls.stream != "image":
            return {"in.data": _IN_DATA}
        return {
            "in.mask": ("Optional region limiter from the slot-1 predecessor, applied to the blended "
                        "result (`mask_blend`)."),
            "in.data": _IN_DATA, "out.image": _OUT_IMAGE}

    def execute(self, image, mask=None, data=None, context=None, samples=None):
        """Single-edge fallback: the one input passes through unchanged."""
        if self.stream != "image" and self.single_edge_combines:
            return self._merge_stream([mask], [samples], [data])
        if self.stream == "mask":
            return {"image": None, "mask": mask}
        if self.stream == "samples":
            return {"image": None, "mask": None, "samples": samples}
        return {"image": image, "mask": mask}

    def execute_multiple(self, images, masks, *, samples=None, datas=None):
        if self.stream != "image":
            return self._merge_stream(masks, samples, datas)
        masks = masks or []
        layers = [(img, masks[i] if i < len(masks) else None)
                  for i, img in enumerate(images or []) if img is not None]
        if not layers:
            return self._null({"image": None, "mask": None})
        if len(layers) < self.multi_input_count:
            return {"image": layers[0][0], "mask": layers[0][1]}
        first = next((d for d in (datas or []) if d is not None), None)
        arg = self._reconciled(layers) if self.reconcile else layers
        res = self._author_fn()(arg, ParamView(self), **self._opt(data=first, datas=datas))
        if isinstance(res, dict):
            return res
        if self.reconcile:
            res = mask_blend(layers[0][0], to_uint8(res), layers[0][1])
        return {"image": res, "mask": layers[0][1]}

    def _merge_stream(self, masks, samples, datas):
        first = next((d for d in (datas or []) if d is not None), None)
        opt = self._opt(data=first, datas=datas)
        if self.stream == "mask":
            valid = [m for m in (masks or []) if m is not None]
            if not valid:
                return self._null({"image": None, "mask": None})
            res = self._author_fn()(valid, ParamView(self), **opt)
            res, extras = _split(res, "mask")
            out = {"image": None, "mask": res}
        else:
            sets = [s for s in (samples or []) if s is not None]
            res = self._author_fn()(sets, ParamView(self), **opt)
            # A samples dict has no "samples" key; an outputs dict does.
            res, extras = _split(res, "samples") if isinstance(res, dict) and "samples" in res                 else (res, {})
            out = {"image": None, "mask": None, "samples": res}
        out.update(extras)
        return out

    @staticmethod
    def _reconciled(layers):
        from PIL import Image
        h, w = layers[0][0].shape[:2]
        out = []
        for i, (img, m) in enumerate(layers):
            if i and img.shape[:2] != (h, w):
                img = np.array(Image.fromarray(img).resize((w, h)))
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            out.append((img, m))
        return out


class SampleConsumer(_Kind):
    """Takes the sample stream (and, by default, an image).

    Author: ``consume(self, image, samples, p[, mask][, data])`` -> dict of
    outputs (or None), merged over the defaults: image and mask carried through
    (image-port nodes only), samples passed on unchanged (``passes_samples``) or
    None (a terminating consumer). Whatever else the dict holds (``data``,
    ``viz``, a replaced ``samples``, a new ``image``) goes out as returned.

    Class attributes: ``passes_samples`` (True, the common case: the node also
    produces samples; False: a terminating consumer, the pipeline clears the
    stream after it, and the default result has no ``samples`` key at all -- a
    node that wants to return samples anyway returns them from ``consume``) and
    ``image_ports`` (True: image in and out; False: a sample-analysis node
    without image ports, ``image`` is None).

    The kind guards ONE case: ``image is None`` on an image-port node returns the
    defaults without calling ``consume``. ``samples`` may be None and may hold no
    islands when ``consume`` runs: the author decides what the null result looks
    like, because hand-written nodes differ in which extra keys (``data``,
    ``viz``) it carries.
    """

    _kind_root = True
    _author = ("consume", 3, frozenset({"mask", "data"}))
    passes_samples = True
    image_ports = True

    @classmethod
    def _flag_defaults(cls):
        d = {"accepts_samples": True, "produces_samples": bool(cls.passes_samples),
             "required_inputs": ("samples",)}
        if cls.passes_samples:
            d["required_outputs"] = ("samples",)
        if not cls.image_ports:
            d.update({"has_input_port": False, "has_output_port": False,
                      "produces_image": False})
        return d

    @classmethod
    def _default_docs(cls):
        d = {"in.data": _IN_DATA,
             "in.samples": _IN_SAMPLES_PASS if cls.passes_samples else _IN_SAMPLES_TERMINAL}
        if cls.passes_samples:
            d["out.samples"] = _OUT_SAMPLES_PASS
        if cls.image_ports:
            d["in.image"] = _IN_IMAGE
            d["out.image"] = _OUT_IMAGE
        return d

    def execute(self, image=None, mask=None, samples=None, data=None, context=None):
        ports = self.image_ports
        out = {"image": image if ports else None,
               "mask": mask if ports and self.accepts_mask else None}
        if self.passes_samples:
            out["samples"] = samples
        if ports and image is None:
            return self._null(out)
        res = self._author_fn()(image if ports else None, samples, ParamView(self),
                                **self._opt(mask=mask if ports or self.accepts_mask else None,
                                            data=data))
        if res is not None:
            out.update(res)
        return out
