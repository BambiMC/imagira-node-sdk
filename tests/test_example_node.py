"""The shipped example must actually work — and it doubles as the template for
what a third-party node's own test file should look like."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from invert_node import ExampleInvertNode  # noqa: E402

from imagira_node_sdk import NODE_REGISTRY, validate_param_schema  # noqa: E402
from imagira_node_sdk.testing import (  # noqa: E402
    assert_node_contract,
    make_image,
    make_mask,
    run_node,
)


def test_contract():
    assert_node_contract(ExampleInvertNode)


def test_schema_is_clean_including_warnings():
    assert validate_param_schema(ExampleInvertNode) == []


def test_the_register_decorator_put_it_in_the_registry():
    assert NODE_REGISTRY["example_invert"] is ExampleInvertNode


def test_inverts_fully_by_default():
    out = run_node(ExampleInvertNode, image=make_image(2, 2, (10, 20, 30)))
    assert out["image"][0, 0].tolist() == [245, 235, 225]


def test_amount_zero_is_a_no_op():
    img = make_image(2, 2, (10, 20, 30))
    out = run_node(ExampleInvertNode, image=img, params={"amount": 0.0})
    assert np.array_equal(out["image"], img)


def test_single_channel_only_touches_that_channel():
    out = run_node(ExampleInvertNode, image=make_image(1, 1, (10, 20, 30)),
                   params={"channels": "green"})
    assert out["image"][0, 0].tolist() == [10, 235, 30]


def test_mask_confines_the_effect():
    img = make_image(1, 2, (10, 10, 10))
    mask = np.array([[True, False]])
    out = run_node(ExampleInvertNode, image=img, mask=mask)
    assert out["image"][0, 0].tolist() == [245, 245, 245]
    assert out["image"][0, 1].tolist() == [10, 10, 10]


def test_mask_is_passed_through_to_downstream_nodes():
    mask = make_mask(2, 2)
    assert np.array_equal(run_node(ExampleInvertNode, image=make_image(2, 2), mask=mask)["mask"],
                          mask)


def test_grayscale_input_is_normalised_rather_than_crashing():
    out = run_node(ExampleInvertNode, image=np.full((2, 2), 10, np.uint8))
    assert out["image"].shape == (2, 2, 3)


def test_a_template_in_the_amount_param_resolves_from_the_data_port():
    # The user typed `{{ $json.strength }}` into the Amount field and wired a
    # data node into this one — nothing in the node's own code knows about it.
    out = run_node(ExampleInvertNode, image=make_image(1, 1, (10, 10, 10)),
                   params={"amount": "{{ $json.strength }}"}, data={"strength": 0.0})
    assert out["image"][0, 0].tolist() == [10, 10, 10]
