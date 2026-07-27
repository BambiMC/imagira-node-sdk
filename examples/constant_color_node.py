"""A **source** node — generates an image instead of transforming one.

Adapted from Imagira's built-in `constant_color`. The interesting part is the
port declaration: a source has no image input at all, so the host app knows not
to demand an upstream connection and not to lint it as disconnected.
"""

from __future__ import annotations

import numpy as np

from imagira_node_sdk import BaseNode, register


@register
class ExampleConstantColorNode(BaseNode):
    type = "example_constant_color"
    label = "Constant Color (example)"
    category = "Image|Source"
    color = "#f59e0b"
    description = "Generates a solid-colour image of a specified size."

    # A source: no image comes in, one goes out. `is_source_node` is what tells
    # the host this node starts a pipeline rather than sitting in the middle of
    # one — without it the app would treat the missing input as an error.
    has_input_port = False
    produces_image = True
    is_source_node = True

    param_schema = [
        {"key": "r", "label": "Red", "type": "slider", "default": 128,
         "min": 0, "max": 255, "help": "Red channel (0–255)."},
        {"key": "g", "label": "Green", "type": "slider", "default": 128,
         "min": 0, "max": 255, "help": "Green channel (0–255)."},
        {"key": "b", "label": "Blue", "type": "slider", "default": 128,
         "min": 0, "max": 255, "help": "Blue channel (0–255)."},
        {"key": "width", "label": "Width", "type": "number", "default": 512,
         "min": 1, "max": 8192, "help": "Width of the generated image in pixels."},
        {"key": "height", "label": "Height", "type": "number", "default": 512,
         "min": 1, "max": 8192, "help": "Height of the generated image in pixels."},
    ]

    # `image` still has a default of None even though this node has no input
    # port — the host may pass one positionally, and a source must not care.
    def execute(self, image=None, mask=None, data=None, context=None):
        p = self.params
        # int() around every read: a param wired to `{{ $json.width }}` resolves
        # to whatever the data payload holds, which may well be a string.
        r = int(p.get("r", 128))
        g = int(p.get("g", 128))
        b = int(p.get("b", 128))
        w = int(p.get("width", 512))
        h = int(p.get("height", 512))
        return {"image": np.full((h, w, 3), [r, g, b], dtype=np.uint8), "mask": None}
