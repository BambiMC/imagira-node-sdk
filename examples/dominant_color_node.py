"""A **samples-producing** node — builds the red wire's island payload.

Adapted from Imagira's built-in `dominant_color`. Use this as the reference for
what a `samples` dict has to contain: the host app and every downstream sampler
node assume this exact shape.
"""

from __future__ import annotations

import numpy as np

from imagira_node_sdk import BaseNode, as_bool_mask, register, to_rgb


@register
class ExampleDominantColorNode(BaseNode):
    type = "example_dominant_color"
    label = "Dominant Color (example)"
    category = "Sample|Sampler"
    color = "#ef4444"
    description = "Finds the most frequently occurring colour in the image or mask region."

    accepts_mask = True
    accepts_data = True
    produces_image = False
    produces_mask = False
    produces_samples = True
    # A terminal node on the blue wire: it emits samples only, so it has no
    # image output port at all.
    has_output_port = False
    required_outputs = ("samples",)

    param_schema = [
        {"key": "num_colors", "label": "Number of Colors", "type": "number",
         "default": 1, "min": 1, "max": 10,
         "help": "How many dominant colour clusters to emit as separate islands."},
        {"key": "use_mask", "label": "Use Mask Region", "type": "bool", "default": False,
         "help": "Restrict the search to pixels inside the mask; when off the whole "
                 "image is analysed."},
        {"key": "bin_count", "label": "Color Bins", "type": "number",
         "default": 32, "min": 4, "max": 128,
         "help": "Quantisation resolution per channel. Higher groups fewer shades "
                 "together: more precise, slower."},
    ]

    def execute(self, image, mask=None, data=None, context=None):
        if image is None:
            return {"image": None, "mask": None,
                    "samples": {"islands": [], "image_size": (0, 0)}}

        img = to_rgb(image)
        h, w = img.shape[:2]
        p = self.params

        # as_bool_mask() first: `mask` legitimately arrives as bool, as uint8
        # (0/1 or 0/255), or as a float soft matte, and only bool works as a
        # selection. np.where on a uint8 mask returns every non-zero pixel,
        # which happens to be right; image[mask] would not be.
        mask_b = as_bool_mask(mask)
        if p.get("use_mask", False) and mask_b is not None and mask_b.any():
            rows, cols = np.nonzero(mask_b)
            pixels = img[rows, cols]
        else:
            rows, cols = np.mgrid[0:h, 0:w]
            rows, cols = rows.ravel(), cols.ravel()
            pixels = img.reshape(-1, 3)
        positions = np.stack([rows, cols], axis=1)

        if pixels.size == 0:
            return {"image": None, "mask": None,
                    "samples": {"islands": [], "image_size": (h, w)}}

        # Quantise into bins, then rank bins by population.
        bins = max(4, int(p.get("bin_count", 32)))
        binned = (pixels // max(1, 256 // bins)).astype(np.int32)
        bin_ids = binned[:, 0] * bins * bins + binned[:, 1] * bins + binned[:, 2]
        unique, counts = np.unique(bin_ids, return_counts=True)
        top = np.argsort(-counts)[:max(1, int(p.get("num_colors", 1)))]

        islands = []
        for rank, idx in enumerate(top):
            member = bin_ids == unique[idx]
            island_pixels = pixels[member]
            islands.append({
                "pixels": island_pixels.astype(np.uint8),      # (N, 3) uint8
                "positions": positions[member].astype(int),    # (N, 2) [row, col]
                # `label` is the stable join key: a downstream node reports its
                # own per-island results against this, so it must not be a bare
                # list index that shifts when islands are filtered out.
                "label": rank + 1,
                "area": int(counts[idx]),
                "mean_color": island_pixels.mean(axis=0).astype(np.uint8).tolist(),
            })

        return {"image": None, "mask": None,
                "samples": {"islands": islands, "image_size": (h, w)}}
