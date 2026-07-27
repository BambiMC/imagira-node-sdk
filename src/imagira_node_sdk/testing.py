"""Test helpers for node authors — the point of the SDK split, in one module.

These let you test a node without installing Imagira at all: instantiate it,
feed it a real numpy image, bind a ``{{ }}`` data context the way the host
pipeline does, and check the return shape.

    from imagira_node_sdk.testing import assert_node_contract, run_node, make_image
    from my_pkg.nodes import MyNode

    def test_contract():
        assert_node_contract(MyNode)

    def test_inverts():
        out = run_node(MyNode, image=make_image(4, 4, (10, 20, 30)))
        assert out["image"][0, 0].tolist() == [245, 235, 225]

Not imported by ``imagira_node_sdk/__init__.py`` — import it explicitly, so it
stays out of the runtime path of a production Imagira instance.
"""

from __future__ import annotations

import contextlib
import inspect

import numpy as np

from .base_node import NODE_REGISTRY, BaseNode
from .expression import reset_current_data, set_current_data
from .schema import ERROR, validate_param_schema

__all__ = [
    "assert_node_contract",
    "run_node",
    "make_image",
    "make_mask",
    "make_samples",
    "isolated_registry",
]

# Keys a node is allowed to return. Extras are almost always a typo ("images",
# "msk") that the pipeline would drop on the floor without a word.
_ALLOWED_RESULT_KEYS = frozenset({"image", "mask", "data", "samples", "_iterations"})


def make_image(height: int = 8, width: int = 8, color=(128, 128, 128)) -> np.ndarray:
    """A uint8 (H, W, 3) RGB image filled with *color*."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def make_mask(height: int = 8, width: int = 8, *, filled: bool = True) -> np.ndarray:
    """A bool (H, W) mask, all-True by default."""
    return np.full((height, width), bool(filled), dtype=np.bool_)


def make_samples(image: np.ndarray, mask: np.ndarray | None = None, *, label: int = 1) -> dict:
    """One-island ``samples`` payload in the documented shape.

    Enough to exercise a node that sets ``accepts_samples``; the host app's
    real sampler produces many islands with meaningful labels.
    """
    if mask is None:
        mask = make_mask(image.shape[0], image.shape[1])
    rows, cols = np.nonzero(mask)
    pixels = image[rows, cols]
    return {
        "islands": [{
            "pixels": pixels.astype(np.uint8),
            "positions": np.stack([rows, cols], axis=-1).astype(int),
            "label": label,
            "area": int(rows.size),
            "mean_color": [int(c) for c in pixels.reshape(-1, 3).mean(axis=0)]
            if rows.size else [0, 0, 0],
        }],
        "image_size": (image.shape[0], image.shape[1]),
    }


@contextlib.contextmanager
def isolated_registry():
    """Run a block with a scratch ``NODE_REGISTRY``, restored on exit.

    Defining a node class inside a test normally leaves it in the process-wide
    registry, which then leaks into every later test. Wrap definitions that use
    ``@register`` in this.
    """
    saved = dict(NODE_REGISTRY)
    try:
        yield NODE_REGISTRY
    finally:
        NODE_REGISTRY.clear()
        NODE_REGISTRY.update(saved)


def assert_node_contract(node_cls, *, allow_schema_warnings: bool = True) -> None:
    """Assert *node_cls* is a well-formed node. Raises ``AssertionError``.

    Collects every problem and reports them together, so you fix a schema once
    rather than once per test run. Checks:

    - it's a concrete ``BaseNode`` subclass with a real ``type``
    - ``type`` looks namespaced (a bare name risks colliding with another
      author's node — first import wins, silently)
    - the port flags and ``required_inputs``/``required_outputs`` agree
    - ``execute()`` is actually overridden, with a compatible signature
    - ``param_schema`` passes :func:`validate_param_schema`
    - ``i18n`` isn't inherited-and-mutated from ``BaseNode``
    """
    problems: list[str] = []

    if not (inspect.isclass(node_cls) and issubclass(node_cls, BaseNode)):
        raise AssertionError(f"{node_cls!r} is not a BaseNode subclass")
    if node_cls is BaseNode:
        raise AssertionError("BaseNode itself is not a node")

    node_type = getattr(node_cls, "type", None)
    if not node_type or node_type == "base":
        problems.append("`type` is unset or still 'base' — the host app's discovery loop "
                        "skips such classes, so the node would never register")
    elif not isinstance(node_type, str):
        problems.append(f"`type` must be a string, got {type(node_type).__name__}")
    else:
        if "type" not in node_cls.__dict__:
            problems.append(f"`type` ({node_type!r}) is inherited, not set on this class — "
                            "the discovery loop skips inherited types")
        if "_" not in node_type:
            problems.append(f"`type` {node_type!r} is not namespaced. NODE_REGISTRY is one flat "
                            "process-wide dict and collisions resolve to first-import-wins "
                            "silently — prefix it, e.g. 'myauthor_{}'".format(node_type))
        if node_type != node_type.lower() or " " in node_type:
            problems.append(f"`type` {node_type!r} should be lowercase with no spaces — it is a "
                            "permanent identifier stored in saved workflow JSON")

    if not getattr(node_cls, "label", None):
        problems.append("`label` is empty — the node palette would show a blank entry")

    # execute()
    if node_cls.execute is BaseNode.execute:
        problems.append("`execute()` is not overridden — BaseNode.execute raises "
                        "NotImplementedError")
    else:
        # The host calls execute(image, mask, **kwargs), adding `samples`,
        # `data` and `context` ONLY for nodes whose accepts_* flag opts in. So
        # the requirement isn't "declare all five" — it's that the flags and the
        # signature agree in both directions. A node with accepts_samples=True
        # and no `samples` parameter is a TypeError on first run; a node with a
        # `data` parameter and accepts_data=False just never receives anything.
        try:
            params = list(inspect.signature(node_cls.execute).parameters.values())[1:]
            names = [p.name for p in params]
            takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
            if not takes_kwargs:
                if "image" not in names:
                    problems.append("`execute()` has no 'image' parameter — the host passes the "
                                    "image positionally to every node.")
                for flag, kwarg in (("accepts_samples", "samples"),
                                    ("accepts_data", "data"),
                                    ("accepts_context", "context")):
                    if getattr(node_cls, flag, False) and kwarg not in names:
                        problems.append(
                            f"{flag} is True but `execute()` has no {kwarg!r} parameter — the "
                            f"host will call execute(..., {kwarg}=...) and raise TypeError. "
                            f"Accept it (default None) or add **kwargs.")
                # The reverse is deliberately NOT checked. Declaring `data=None`
                # / `context=None` without the flag is the canonical signature
                # BaseNode.execute itself uses, and nearly every built-in node
                # copies it; the parameter simply stays None. Harmless, and
                # flagging it would fire on almost every correct node.
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            pass

    # port flags vs required_* (same rule __init_subclass__ enforces, restated
    # here so a test failure names every problem rather than the first import
    # error, and so it also covers the accepts/produces pairs it can't see)
    for port, flag in (("mask", "accepts_mask"), ("samples", "accepts_samples"),
                       ("data", "accepts_data"), ("image", "has_input_port")):
        if port in getattr(node_cls, "required_inputs", ()) and not getattr(node_cls, flag, False):
            problems.append(f"required_inputs includes {port!r} but {flag} is False")
    for port, flag in (("mask", "produces_mask"), ("samples", "produces_samples"),
                       ("data", "produces_data"), ("image", "has_output_port")):
        if port in getattr(node_cls, "required_outputs", ()) and not getattr(node_cls, flag, False):
            problems.append(f"required_outputs includes {port!r} but {flag} is False")
    if getattr(node_cls, "produces_image", False) and not getattr(node_cls, "has_output_port", False):
        problems.append("produces_image is True but has_output_port is False — the image has "
                        "nowhere to go")
    if getattr(node_cls, "is_sink_node", False) and getattr(node_cls, "has_output_port", False):
        problems.append("is_sink_node is True but has_output_port is also True — a sink is "
                        "terminal by definition")
    if getattr(node_cls, "is_source_node", False) and getattr(node_cls, "has_input_port", False):
        problems.append("is_source_node is True but has_input_port is also True — a source "
                        "generates its own image rather than taking one in")

    if getattr(node_cls, "supports_gpu", False) and node_cls.execute_gpu is BaseNode.execute_gpu:
        problems.append("supports_gpu is True but execute_gpu() is not overridden — the node "
                        "would silently run on the CPU path anyway")

    # i18n must not be the shared BaseNode dict if it has content
    if node_cls.i18n and "i18n" not in node_cls.__dict__:
        problems.append("i18n has content but is inherited — it is a shared class-level dict, so "
                        "mutating it leaks translations onto every other node. Define your own "
                        "`i18n = {...}`.")

    for issue in validate_param_schema(node_cls):
        if issue.level == ERROR or not allow_schema_warnings:
            problems.append(str(issue))

    if problems:
        raise AssertionError(
            f"{node_cls.__name__} violates the node contract:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )


def run_node(
    node_cls,
    *,
    image=None,
    mask=None,
    samples=None,
    data=None,
    context=None,
    params=None,
    node_id: str = "test_node",
    all_nodes=None,
    check_result: bool = True,
):
    """Instantiate *node_cls*, run ``execute()`` the way the host does, return
    the result dict.

    Mirrors the host's calling convention exactly: ``image`` and ``mask`` go
    positionally, while ``samples``, ``data`` and ``context`` are passed **only
    when the node's matching ``accepts_*`` flag is True**. That's the real
    contract — a node without ``accepts_data`` never sees a data payload no
    matter what its signature says — so a test that called every node with all
    five would pass for nodes the app would break on.

    Also binds *data* / *all_nodes* as the ``{{ }}`` template context for the
    duration of the call, so a param set to ``"{{ $json.name }}"`` resolves here
    just as it would in the app. Without that binding your test would see the raw
    template text and pass for the wrong reason.

    With ``check_result`` (default), the returned dict is checked against the
    documented contract: allowed keys only, ``image`` uint8 (H, W, 3), ``mask``
    boolean (H, W) matching the image's size.
    """
    node = node_cls(node_id, params=params or {})
    if image is None and getattr(node_cls, "has_input_port", True):
        image = make_image()

    kwargs = {}
    if getattr(node_cls, "accepts_samples", False):
        kwargs["samples"] = samples
    if getattr(node_cls, "accepts_data", False):
        kwargs["data"] = data
    if getattr(node_cls, "accepts_context", False) or context is not None:
        # accepts_context is a host-side duck-typed flag rather than a BaseNode
        # attribute (see docs/authoring.md); an explicit context= in a test is
        # taken as intent to pass one regardless.
        kwargs["context"] = context

    # The data-port payload is bound as the template source even for nodes that
    # don't declare accepts_data — the host binds it around every execute().
    token = set_current_data(data, all_nodes or {})
    try:
        result = node.execute(image, mask, **kwargs)
    finally:
        reset_current_data(token)

    if check_result:
        _check_result(node_cls, result)
    return result


def _check_result(node_cls, result) -> None:
    name = node_cls.__name__
    if result is None:
        raise AssertionError(
            f"{name}.execute() returned None. It must return a dict, at minimum "
            f'{{"image": ...}} — the pipeline reads the next node\'s input out of it.')
    if not isinstance(result, dict):
        raise AssertionError(f"{name}.execute() must return a dict, got {type(result).__name__}")

    unknown = sorted(set(result) - _ALLOWED_RESULT_KEYS)
    if unknown:
        raise AssertionError(
            f"{name}.execute() returned unrecognised key(s) {unknown}; the pipeline reads only "
            f"{sorted(_ALLOWED_RESULT_KEYS)} and drops anything else silently")

    img = result.get("image")
    if img is not None:
        if not isinstance(img, np.ndarray):
            raise AssertionError(f'{name}: result["image"] must be a numpy array, got '
                                 f"{type(img).__name__}")
        if img.dtype != np.uint8:
            raise AssertionError(f'{name}: result["image"] must be uint8, got {img.dtype}. '
                                 f"A float image silently wraps or clips downstream — finish with "
                                 f"np.clip(...).astype(np.uint8), or run it through to_rgb().")
        if img.ndim != 3 or img.shape[2] != 3:
            raise AssertionError(f'{name}: result["image"] must be (H, W, 3) RGB, got '
                                 f"{img.shape}. to_rgb() normalises grayscale/RGBA for you.")
    elif getattr(node_cls, "produces_image", False) and not getattr(node_cls, "is_sink_node", False):
        raise AssertionError(f'{name}: produces_image is True but result["image"] is None')

    m = result.get("mask")
    if m is not None:
        if not isinstance(m, np.ndarray):
            raise AssertionError(f'{name}: result["mask"] must be a numpy array, got '
                                 f"{type(m).__name__}")
        if m.dtype != np.bool_:
            raise AssertionError(f'{name}: result["mask"] must be np.bool_, got {m.dtype}. '
                                 f"A uint8 0/255 mask is truthy everywhere it isn't 0, which "
                                 f"quietly turns a soft mask into a hard one.")
        if m.ndim != 2:
            raise AssertionError(f'{name}: result["mask"] must be 2-D (H, W), got {m.shape}')
        if img is not None and m.shape != img.shape[:2]:
            raise AssertionError(f'{name}: mask shape {m.shape} does not match image '
                                 f"{img.shape[:2]}")

    d = result.get("data")
    if d is not None and not isinstance(d, dict):
        raise AssertionError(f'{name}: result["data"] must be a dict or None, got '
                             f"{type(d).__name__}")

    s = result.get("samples")
    if s is not None:
        if not isinstance(s, dict):
            raise AssertionError(f'{name}: result["samples"] must be a dict, got '
                                 f"{type(s).__name__}")
        if "islands" not in s:
            raise AssertionError(f'{name}: result["samples"] is missing "islands" — the '
                                 f'documented shape is {{"islands": [...], "image_size": (H, W)}}')
