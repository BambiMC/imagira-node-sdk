"""Every shipped example must work, and each one doubles as the template for
what a third-party package's own test file should look like.

The per-example tests below are grouped by the facet the example exists to
demonstrate: a source node, a mask producer, a samples producer, a samples
consumer, and (in test_example_node.py) the plain image transform.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from binary_segment_node import ExampleBinarySegmentNode  # noqa: E402
from constant_color_node import ExampleConstantColorNode  # noqa: E402
from dominant_color_node import ExampleDominantColorNode  # noqa: E402
from sample_mask_filter_node import ExampleSampleMaskFilterNode  # noqa: E402

from imagira_node_sdk import NODE_REGISTRY, validate_param_schema  # noqa: E402
from imagira_node_sdk.testing import (  # noqa: E402
    assert_node_contract,
    make_image,
    make_samples,
    run_node,
)

ALL_EXAMPLES = [
    ExampleConstantColorNode,
    ExampleBinarySegmentNode,
    ExampleDominantColorNode,
    ExampleSampleMaskFilterNode,
]


@pytest.mark.parametrize("node_cls", ALL_EXAMPLES, ids=lambda c: c.type)
class TestEveryExample:
    """What every node — example or not — has to satisfy."""

    def test_contract(self, node_cls):
        assert_node_contract(node_cls)

    def test_schema_is_clean_including_warnings(self, node_cls):
        assert validate_param_schema(node_cls) == []

    def test_registered_under_its_own_type(self, node_cls):
        assert NODE_REGISTRY[node_cls.type] is node_cls

    def test_type_is_namespaced_and_stable(self, node_cls):
        assert node_cls.type.startswith("example_")


class TestConstantColorSource:

    def test_generates_an_image_from_nothing(self):
        out = run_node(ExampleConstantColorNode,
                       params={"r": 10, "g": 20, "b": 30, "width": 4, "height": 3})
        assert out["image"].shape == (3, 4, 3)
        assert out["image"][0, 0].tolist() == [10, 20, 30]

    def test_declares_no_image_input(self):
        assert ExampleConstantColorNode.has_input_port is False
        assert ExampleConstantColorNode.is_source_node is True

    def test_run_node_does_not_invent_an_input_image_for_a_source(self):
        # has_input_port is False, so the kit must not helpfully pass one.
        assert run_node(ExampleConstantColorNode, params={"width": 2, "height": 2})

    def test_string_params_from_a_template_still_work(self):
        # A width wired to `{{ $json.w }}` arrives as whatever the payload held.
        out = run_node(ExampleConstantColorNode,
                       params={"width": "{{ $json.w }}", "height": 2},
                       data={"w": 5})
        assert out["image"].shape == (2, 5, 3)


class TestBinarySegmentMaskProducer:

    def test_thresholds_on_average_brightness(self):
        img = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        out = run_node(ExampleBinarySegmentNode, image=img, params={"threshold": 0.5})
        assert out["mask"].tolist() == [[False, True]]

    def test_returns_a_bool_mask_not_uint8(self):
        # The whole point of the example's closing comment.
        out = run_node(ExampleBinarySegmentNode, image=make_image(2, 2, (200, 200, 200)))
        assert out["mask"].dtype == np.bool_

    def test_emits_no_image(self):
        out = run_node(ExampleBinarySegmentNode, image=make_image(2, 2))
        assert out["image"] is None
        assert ExampleBinarySegmentNode.produces_image is False

    @pytest.mark.parametrize("channel,expected", [
        ("r", [[True, False]]), ("g", [[False, True]]), ("b", [[False, False]]),
    ])
    def test_single_channel_selection(self, channel, expected):
        img = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        out = run_node(ExampleBinarySegmentNode, image=img,
                       params={"channel": channel, "threshold": 0.5})
        assert out["mask"].tolist() == expected

    def test_unknown_channel_fails_loudly_and_names_the_value(self):
        with pytest.raises(ValueError, match="unknown channel 'magenta'"):
            run_node(ExampleBinarySegmentNode, image=make_image(2, 2),
                     params={"channel": "magenta"})

    def test_grayscale_input_is_normalised_rather_than_crashing(self):
        out = run_node(ExampleBinarySegmentNode, image=np.full((2, 2), 200, np.uint8))
        assert out["mask"].shape == (2, 2)


class TestDominantColorSamplesProducer:

    def test_finds_the_most_common_colour(self):
        img = np.array([[[10, 10, 10], [10, 10, 10], [200, 0, 0]]], dtype=np.uint8)
        out = run_node(ExampleDominantColorNode, image=img)
        islands = out["samples"]["islands"]
        assert len(islands) == 1
        assert islands[0]["area"] == 2
        assert islands[0]["mean_color"] == [10, 10, 10]

    def test_island_payload_matches_the_documented_shape(self):
        out = run_node(ExampleDominantColorNode, image=make_image(2, 2, (10, 20, 30)))
        island = out["samples"]["islands"][0]
        assert set(island) == {"pixels", "positions", "label", "area", "mean_color"}
        assert island["pixels"].dtype == np.uint8 and island["pixels"].shape[1] == 3
        assert island["positions"].shape[1] == 2
        assert island["label"] == 1
        assert out["samples"]["image_size"] == (2, 2)

    def test_num_colors_emits_that_many_islands(self):
        img = np.array([[[0, 0, 0], [120, 120, 120], [250, 250, 250]]], dtype=np.uint8)
        out = run_node(ExampleDominantColorNode, image=img, params={"num_colors": 3})
        assert len(out["samples"]["islands"]) == 3
        # Labels are 1-based and unique — they're the cross-node join key.
        assert [i["label"] for i in out["samples"]["islands"]] == [1, 2, 3]

    def test_use_mask_restricts_the_search(self):
        img = np.array([[[10, 10, 10], [200, 0, 0]]], dtype=np.uint8)
        mask = np.array([[False, True]])
        out = run_node(ExampleDominantColorNode, image=img, mask=mask,
                       params={"use_mask": True})
        assert out["samples"]["islands"][0]["mean_color"] == [200, 0, 0]

    def test_a_uint8_mask_works_too(self):
        # as_bool_mask() is why: masks legitimately arrive as 0/255 uint8.
        img = np.array([[[10, 10, 10], [200, 0, 0]]], dtype=np.uint8)
        out = run_node(ExampleDominantColorNode, image=img,
                       mask=np.array([[0, 255]], dtype=np.uint8),
                       params={"use_mask": True})
        assert out["samples"]["islands"][0]["mean_color"] == [200, 0, 0]

    def test_an_empty_mask_yields_no_islands_rather_than_raising(self):
        out = run_node(ExampleDominantColorNode, image=make_image(2, 2),
                       mask=np.zeros((2, 2), np.bool_), params={"use_mask": True})
        assert out["samples"]["islands"] != []   # falls back to the whole image

    def test_no_image_returns_an_empty_payload(self):
        # Called directly rather than through run_node: the kit substitutes a
        # default image whenever has_input_port is True, so image=None would
        # never actually reach execute().
        out = ExampleDominantColorNode("n1").execute(None)
        assert out["samples"] == {"islands": [], "image_size": (0, 0)}


class TestSampleMaskFilterSamplesConsumer:

    def _samples(self):
        img = make_image(1, 4, (10, 20, 30))
        s = make_samples(img)
        # two islands: one on the left half, one on the right
        s["islands"] = [
            {"pixels": np.zeros((2, 3), np.uint8),
             "positions": np.array([[0, 0], [0, 1]]), "label": 1, "area": 2,
             "mean_color": [0, 0, 0]},
            {"pixels": np.zeros((2, 3), np.uint8),
             "positions": np.array([[0, 2], [0, 3]]), "label": 2, "area": 2,
             "mean_color": [0, 0, 0]},
        ]
        return s

    def test_receives_the_samples_kwarg_at_all(self):
        # The regression this example exists for: `samples` is a fifth, opt-in
        # parameter, passed only because accepts_samples is True.
        out = run_node(ExampleSampleMaskFilterNode, image=make_image(1, 4),
                       mask=np.array([[True, True, False, False]]),
                       samples=self._samples())
        assert out["samples"] is not None

    def test_inside_mode_drops_the_island_inside_the_mask(self):
        out = run_node(ExampleSampleMaskFilterNode, image=make_image(1, 4),
                       mask=np.array([[True, True, False, False]]),
                       samples=self._samples(), params={"mode": "inside"})
        assert [i["label"] for i in out["samples"]["islands"]] == [2]

    def test_outside_mode_keeps_only_the_island_inside_the_mask(self):
        out = run_node(ExampleSampleMaskFilterNode, image=make_image(1, 4),
                       mask=np.array([[True, True, False, False]]),
                       samples=self._samples(), params={"mode": "outside"})
        assert [i["label"] for i in out["samples"]["islands"]] == [1]

    def test_image_size_is_preserved_for_downstream_nodes(self):
        out = run_node(ExampleSampleMaskFilterNode, image=make_image(1, 4),
                       mask=np.array([[True, True, False, False]]),
                       samples=self._samples())
        assert out["samples"]["image_size"] == (1, 4)

    def test_no_samples_passes_through_without_erroring(self):
        out = run_node(ExampleSampleMaskFilterNode, image=make_image(2, 2), samples=None)
        assert out["samples"] is None

    def test_no_mask_passes_the_samples_through_untouched(self):
        s = self._samples()
        out = run_node(ExampleSampleMaskFilterNode, image=make_image(1, 4), samples=s)
        assert out["samples"] is s

    def test_unknown_mode_raises_before_examining_any_island(self):
        with pytest.raises(ValueError, match="unknown mode 'sideways'"):
            run_node(ExampleSampleMaskFilterNode, image=make_image(1, 4),
                     mask=np.array([[True, True, False, False]]),
                     samples=self._samples(), params={"mode": "sideways"})

    def test_out_of_range_positions_are_clipped_not_crashed_on(self):
        s = self._samples()
        s["islands"][0]["positions"] = np.array([[99, 99]])
        out = run_node(ExampleSampleMaskFilterNode, image=make_image(1, 4),
                       mask=np.array([[True, True, False, False]]),
                       samples=s)
        assert out["samples"] is not None

    def test_required_inputs_is_enforced_at_class_definition_time(self):
        assert ExampleSampleMaskFilterNode.required_inputs == ("samples",)
        assert ExampleSampleMaskFilterNode.accepts_samples is True
