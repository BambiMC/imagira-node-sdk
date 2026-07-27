"""Imagira Node SDK — the plugin contract for third-party Imagira nodes.

Note: "Node" here means *graph node* (a step in an image-processing pipeline).
This is a Python package, not a Node.js one.

Write a node by subclassing :class:`BaseNode` and implementing ``execute()``::

    from imagira_node_sdk import BaseNode, register, to_rgb

    @register
    class InvertNode(BaseNode):
        type        = "myauthor_invert"      # permanent id — prefix it!
        label       = "Invert"
        category    = "Post-Process"
        description = "Inverts every channel."
        param_schema = [
            {"key": "amount", "label": "Amount", "type": "slider",
             "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
        ]

        def execute(self, image, mask=None, data=None, context=None):
            amount = self.params.get("amount", 1.0)
            out = to_rgb(image).copy()
            out = (255 - out) * amount + out * (1 - amount)
            return {"image": out.astype("uint8"), "mask": mask}

Then publish it under the ``imagira.nodes`` entry-point group so a running
Imagira instance discovers it — see ``docs/packaging.md``.

Everything in ``__all__`` below is covered by the semver promise in
``docs/versioning.md``. Anything else in this package is an implementation
detail and may change in a patch release.
"""

from __future__ import annotations

from .base_node import (
    NODE_REGISTRY,
    BaseNode,
    ComputeBackend,
    as_bool_mask,
    mask_blend,
    register,
    to_rgb,
)
from .expression import (
    ExpressionError,
    ExprParams,
    evaluate,
    make_env,
    reset_current_data,
    resolve_template,
    set_current_data,
)
from .schema import (
    CANONICAL_PARAM_TYPES,
    SchemaIssue,
    validate_param_schema,
)

try:  # pragma: no cover - trivial, and only the fallback path is untested
    from importlib.metadata import version

    __version__ = version("imagira-node-sdk")
except Exception:  # not installed (e.g. running straight from a source tree)
    __version__ = "0.0.0.dev0"

__all__ = [
    # ── the contract ────────────────────────────────────────────────────────
    "BaseNode",
    "register",
    "NODE_REGISTRY",
    "ComputeBackend",
    # ── image helpers every node wants ──────────────────────────────────────
    "to_rgb",
    "mask_blend",
    "as_bool_mask",
    # ── params & templating ─────────────────────────────────────────────────
    "ExprParams",
    "ExpressionError",
    "resolve_template",
    # ── host-app surface: binding the {{ }} data context around execute() ──
    "set_current_data",
    "reset_current_data",
    # ── host-app surface: the control-flow condition evaluator ─────────────
    "evaluate",
    "make_env",
    # ── param_schema tooling (opt-in; call it from your own tests) ─────────
    "validate_param_schema",
    "SchemaIssue",
    "CANONICAL_PARAM_TYPES",
    "__version__",
]
