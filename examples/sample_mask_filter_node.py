"""A **samples-consuming** node — and the one example that shows `samples`
arriving as an argument.

Adapted from Imagira's built-in `sample_mask_filter`. Two things worth copying:

1. `samples` is a **fifth, opt-in parameter** of `execute()`. The host passes it
   only when the node sets `accepts_samples = True`, so it isn't in the
   four-argument signature most nodes use. Same story for `data`
   (`accepts_data`) and `context` (`accepts_context`) — see docs/authoring.md.
2. `required_inputs = ("samples",)` says the node is useless without them. The
   editor draws a filled port dot, lint errors if it's disconnected, and
   `__init_subclass__` would reject the class outright if `accepts_samples`
   weren't also True.
"""

from __future__ import annotations

import numpy as np

from imagira_node_sdk import BaseNode, as_bool_mask, register


@register
class ExampleSampleMaskFilterNode(BaseNode):
    type = "example_sample_mask_filter"
    label = "Filter Samples by Mask (example)"
    category = "Sample|Filter"
    color = "#ef4444"
    description = (
        "Removes sample islands whose pixels overlap a mask region.\n"
        "• inside — drop islands mostly inside the mask\n"
        "• outside — drop islands mostly outside the mask"
    )

    has_input_port = True
    has_output_port = False
    accepts_mask = True
    accepts_samples = True
    produces_samples = True
    produces_image = False
    required_inputs = ("samples",)
    required_outputs = ("samples",)

    param_schema = [
        {"key": "mode", "label": "Remove islands that are", "type": "select",
         "default": "inside", "options": ["inside", "outside"],
         "help": "inside — remove islands mostly inside the mask\n"
                 "outside — remove islands mostly outside the mask"},
        {"key": "threshold", "label": "Overlap threshold (%)", "type": "slider",
         "default": 50, "min": 1, "max": 100,
         "help": "An island counts as 'inside' when at least this percentage of "
                 "its sampled pixels fall within the mask."},
    ]

    def execute(self, image=None, mask=None, samples=None, data=None, context=None):
        # Nothing to filter is not an error — pass the graph's state through
        # unchanged so the rest of the pipeline keeps running.
        if samples is None:
            return {"image": image, "mask": mask, "samples": None}
        islands = samples.get("islands") or []
        if not islands or mask is None:
            return {"image": image, "mask": mask, "samples": samples}

        mask_b = as_bool_mask(mask)
        h, w = mask_b.shape[:2]
        mode = self.params.get("mode", "inside")
        threshold = float(self.params.get("threshold", 50)) / 100.0

        # Validate before the loop, not inside it: a bad `mode` is a
        # configuration error that should fail on the first island, not on
        # whichever one happens to take the other branch.
        if mode not in ("inside", "outside"):
            raise ValueError(f"{self.type}: unknown mode {mode!r}")

        kept = []
        for island in islands:
            positions = island.get("positions")
            if positions is None or len(positions) == 0:
                kept.append(island)
                continue
            pos = np.asarray(positions, dtype=np.int32)
            # Clip rather than trust: an island may carry positions sampled from
            # a larger image earlier in the graph, and an out-of-range index
            # would raise here rather than in whoever produced it.
            rows = np.clip(pos[:, 0], 0, h - 1)
            cols = np.clip(pos[:, 1], 0, w - 1)
            is_inside = float(mask_b[rows, cols].mean()) >= threshold
            # Keep the ones the mode does NOT target: mode="inside" removes
            # inside islands, so it keeps the outside ones.
            if (mode == "inside") != is_inside:
                kept.append(island)

        return {
            "image": image,
            "mask": mask,
            # Preserve image_size — downstream nodes map positions back with it.
            "samples": {"islands": kept, "image_size": samples.get("image_size")},
        }
