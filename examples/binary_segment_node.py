"""A **mask-producing** node — its product is the orange wire, not the blue one.

Adapted from Imagira's built-in `binary_segment`. Shows the flag combination for
a node whose primary output is a mask, and the reason
`required_outputs = ("mask",)` exists.
"""

from __future__ import annotations

import numpy as np

from imagira_node_sdk import BaseNode, register, to_rgb


@register
class ExampleBinarySegmentNode(BaseNode):
    type = "example_binary_segment"
    label = "Binary Segment (example)"
    category = "Mask|Source"
    color = "#e74c3c"
    description = "Performs binary segmentation using a fixed threshold on intensity."

    accepts_mask = True
    accepts_data = True
    # The node consumes an image but doesn't emit one — its product is a mask.
    # `required_outputs = ("mask",)` marks that as the primary output, which
    # gives a filled port dot in the editor and a lint error if it's left
    # dangling. Declaring it while `produces_mask` was False would raise
    # TypeError the moment this module is imported.
    produces_image = False
    produces_mask = True
    required_outputs = ("mask",)

    param_schema = [
        {"key": "threshold", "label": "Threshold", "type": "slider",
         "default": 0.5, "min": 0, "max": 1, "step": 0.01,
         "help": "Normalised brightness cutoff (0–1); pixels above this value are "
                 "kept. Increase to keep only brighter areas."},
        {"key": "channel", "label": "Channel", "type": "select", "default": "all",
         "options": ["r", "g", "b", "all"],
         "help": "Colour channel evaluated against the threshold.\n"
                 "• r — red channel only\n• g — green channel only\n"
                 "• b — blue channel only\n• all — average of all three"},
    ]

    def execute(self, image, mask=None, data=None, context=None):
        if image is None:
            return {"image": None, "mask": mask}

        p = self.params
        img_f = to_rgb(image).astype(np.float32) / 255.0

        channel = p.get("channel", "all")
        if channel == "all":
            gray = img_f.mean(axis=2)
        elif channel in ("r", "g", "b"):
            gray = img_f[:, :, "rgb".index(channel)]
        else:
            # Fail loudly and name the value. A silent fallback to "all" here
            # would look like the threshold was wrong, not the channel.
            raise ValueError(f"{self.type}: unknown channel {channel!r}")

        # Return np.bool_, not uint8. A 0/1 uint8 mask *looks* fine and even
        # blends correctly, but any downstream node that uses it to index
        # (`image[mask]`) gets integer indexing instead of a selection —
        # silently wrong pixels, or an IndexError if the mask holds 255s.
        result = gray > float(p.get("threshold", 0.5))
        return {"image": None, "mask": result.astype(np.bool_)}
