"""A complete, working node — copy this as a starting point.

Depends on nothing but ``imagira_node_sdk`` and numpy. Run its tests with
``pytest examples/`` and it never touches an Imagira install.
"""

from __future__ import annotations

import numpy as np

from imagira_node_sdk import BaseNode, mask_blend, register, to_rgb


@register
class ExampleInvertNode(BaseNode):
    # `type` is a PERMANENT identifier: saved workflow JSON references nodes by
    # this string, so renaming it breaks every workflow that used the node.
    # Prefix it with your author/package name — NODE_REGISTRY is one flat
    # process-wide dict and collisions resolve to first-import-wins, silently.
    type = "example_invert"
    label = "Invert (example)"
    category = "Post-Process"
    color = "#7c5cff"
    description = "Inverts the image, optionally only inside a mask."

    # Ports. The defaults already give one image in and one image out, so only
    # the mask flag has to be declared here.
    accepts_mask = True

    param_schema = [
        {
            "key": "amount", "label": "Amount", "type": "slider",
            "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "help": "0 = original, 1 = fully inverted.",
        },
        {
            "key": "channels", "label": "Channels", "type": "select",
            "default": "all", "options": ["all", "red", "green", "blue"],
        },
    ]

    # Optional translations. Note the rule: define your OWN dict, never mutate
    # the inherited one — it's a shared class-level object.
    i18n = {
        "de": {
            "label": "Invertieren (Beispiel)",
            "description": "Invertiert das Bild, optional nur innerhalb einer Maske.",
            "params": {"amount": {"label": "Stärke"}},
        },
    }

    def execute(self, image, mask=None, data=None, context=None):
        # Always read params through self.params.get(). It's an ExprParams, so
        # this is also what resolves a `{{ $json.amount }}` template a user may
        # have typed into the field. Reading self.params directly (or via
        # .items()) bypasses that and hands you the raw template text.
        amount = float(self.params.get("amount", 1.0))
        channels = self.params.get("channels", "all")

        # to_rgb() normalises grayscale/RGBA/float input to uint8 (H, W, 3), so
        # the node doesn't have to care what the upstream node produced.
        src = to_rgb(image)
        out = src.astype(np.float32)

        inverted = 255.0 - out
        if channels == "all":
            out = inverted * amount + out * (1.0 - amount)
        else:
            idx = {"red": 0, "green": 1, "blue": 2}[channels]
            out[:, :, idx] = inverted[:, :, idx] * amount + out[:, :, idx] * (1.0 - amount)

        # Finish in uint8 — a float image silently wraps or clips downstream.
        result = np.clip(out, 0, 255).astype(np.uint8)

        # mask_blend() confines the effect to the mask, and returns `result`
        # untouched when mask is None.
        result = mask_blend(src, result, mask)

        # The pipeline reads the next node's input out of this dict. Pass the
        # mask through so downstream nodes still see it.
        return {"image": result, "mask": mask}
