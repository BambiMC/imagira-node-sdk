"""The contract itself: class-definition-time validation, defaults caching,
serialization, the registry, and the two image helpers.

These are the tests that have to keep passing byte-for-byte in behaviour, since
every third-party node in existence depends on them.
"""

from __future__ import annotations

import numpy as np
import pytest

from imagira_node_sdk import (
    as_bool_mask,
    NODE_REGISTRY,
    BaseNode,
    ComputeBackend,
    mask_blend,
    register,
    set_current_data,
    reset_current_data,
    to_rgb,
)
from imagira_node_sdk.testing import isolated_registry


def _node(**attrs):
    """Define a throwaway node subclass with *attrs* set on it."""
    attrs.setdefault("type", "t_" + "_".join(sorted(attrs)) or "t")
    return type("TmpNode", (BaseNode,), attrs)


class TestInitSubclassValidation:
    """Misconfigured nodes must fail at import, not mid-pipeline."""

    @pytest.mark.parametrize("port,flag", [
        ("mask", "accepts_mask"),
        ("samples", "accepts_samples"),
        ("data", "accepts_data"),
        ("image", "has_input_port"),
    ])
    def test_required_input_without_accepting_flag_raises(self, port, flag):
        with pytest.raises(TypeError) as exc:
            _node(type="v_in_" + port, required_inputs=(port,), **{flag: False})
        assert port in str(exc.value)
        assert flag in str(exc.value)
        assert "must also be accepted" in str(exc.value)

    @pytest.mark.parametrize("port,flag", [
        ("mask", "produces_mask"),
        ("samples", "produces_samples"),
        ("data", "produces_data"),
        ("image", "has_output_port"),
    ])
    def test_required_output_without_producing_flag_raises(self, port, flag):
        with pytest.raises(TypeError) as exc:
            _node(type="v_out_" + port, required_outputs=(port,), **{flag: False})
        assert port in str(exc.value)
        assert flag in str(exc.value)
        assert "must also be produced" in str(exc.value)

    @pytest.mark.parametrize("port,flag", [
        ("mask", "accepts_mask"),
        ("samples", "accepts_samples"),
        ("data", "accepts_data"),
        ("image", "has_input_port"),
    ])
    def test_required_input_group_member_without_accepting_flag_raises(self, port, flag):
        # An OR-group has the same rule as required_inputs: every port named in
        # it must be one the node actually accepts, or the group can never be
        # satisfied through that port.
        other = "data" if port != "data" else "mask"
        # Accept everything, then switch off just the flag under test — built as
        # one dict so the override can't collide with the defaults.
        attrs = {"type": "g_in_" + port, "required_input_groups": ((port, other),),
                 "accepts_mask": True, "accepts_samples": True, "accepts_data": True}
        attrs[flag] = False
        with pytest.raises(TypeError) as exc:
            _node(**attrs)
        assert "required_input_groups" in str(exc.value)
        assert flag in str(exc.value)

    def test_single_port_group_is_rejected_as_a_misuse(self):
        with pytest.raises(TypeError) as exc:
            _node(type="g_single", accepts_mask=True, required_input_groups=(("mask",),))
        assert "at least 2 ports" in str(exc.value)
        assert "belongs in required_inputs" in str(exc.value)

    def test_a_valid_or_group_is_accepted(self):
        cls = _node(type="g_ok", accepts_mask=True, accepts_samples=True,
                    required_input_groups=(("mask", "samples"),))
        assert cls.required_input_groups == (("mask", "samples"),)

    def test_or_group_members_are_not_individually_required(self):
        # The whole point: neither port is in required_inputs, so wiring either
        # one satisfies the node.
        cls = _node(type="g_notreq", accepts_mask=True, accepts_samples=True,
                    required_input_groups=(("mask", "samples"),))
        assert cls.required_inputs == ()

    def test_default_is_an_empty_tuple(self):
        assert BaseNode.required_input_groups == ()

    def test_consistent_declaration_is_accepted(self):
        cls = _node(type="v_ok", accepts_mask=True, required_inputs=("mask",),
                    produces_mask=True, required_outputs=("mask",))
        assert cls.required_inputs == ("mask",)

    def test_validation_is_skipped_for_abstract_subclasses(self):
        # A shared intermediate base that doesn't declare its own type is a
        # normal pattern; it must not be validated (or it would have to
        # duplicate flags it exists to factor out).
        cls = type("AbstractMixin", (BaseNode,), {"required_inputs": ("mask",)})
        assert cls.type == "base"

    def test_unknown_port_name_in_required_inputs_is_ignored(self):
        # Only the four known streams are checked; an unknown name is not
        # rejected (forward-compatible with a future port type).
        cls = _node(type="v_unknown_port", required_inputs=("nonexistent",))
        assert cls.required_inputs == ("nonexistent",)

    def test_error_fires_at_class_definition_not_instantiation(self):
        # The whole value of the check: it happens on import.
        raised = False
        try:
            _node(type="v_timing", required_inputs=("mask",), accepts_mask=False)
        except TypeError:
            raised = True
        assert raised, "validation must fire while the class body is being defined"


class TestDefaultParams:

    def test_defaults_are_cached_from_param_schema(self):
        cls = _node(type="d_cache", param_schema=[
            {"key": "a", "type": "number", "default": 1},
            {"key": "b", "type": "text", "default": "x"},
        ])
        assert cls._DEFAULT_PARAMS == {"a": 1, "b": "x"}

    def test_missing_default_becomes_none(self):
        cls = _node(type="d_none", param_schema=[{"key": "a", "type": "text"}])
        assert cls._DEFAULT_PARAMS == {"a": None}

    def test_schema_entry_without_key_raises_at_definition_time(self):
        with pytest.raises(KeyError):
            _node(type="d_nokey", param_schema=[{"label": "oops", "type": "text"}])

    def test_each_subclass_gets_its_own_defaults(self):
        a = _node(type="d_a", param_schema=[{"key": "a", "type": "text", "default": 1}])
        b = _node(type="d_b", param_schema=[{"key": "b", "type": "text", "default": 2}])
        assert a._DEFAULT_PARAMS == {"a": 1}
        assert b._DEFAULT_PARAMS == {"b": 2}

    def test_instance_params_start_from_defaults_and_are_overridable(self):
        cls = _node(type="d_inst", param_schema=[
            {"key": "a", "type": "number", "default": 1},
            {"key": "b", "type": "number", "default": 2},
        ])
        node = cls("n1", params={"b": 99})
        assert node.params.get("a") == 1
        assert node.params.get("b") == 99

    def test_mutating_one_instance_does_not_affect_the_class_defaults(self):
        cls = _node(type="d_isolate", param_schema=[{"key": "a", "type": "number", "default": 1}])
        n1, n2 = cls("n1"), cls("n2")
        n1.set_param_value("a", 42)
        assert n2.params.get("a") == 1
        assert cls._DEFAULT_PARAMS == {"a": 1}

    def test_default_params_method_kept_for_backwards_compat(self):
        cls = _node(type="d_legacy", param_schema=[{"key": "a", "type": "number", "default": 1}])
        node = cls("n1")
        assert node._default_params() == {"a": 1}
        # must be a copy, not the shared class dict
        node._default_params()["a"] = 999
        assert cls._DEFAULT_PARAMS == {"a": 1}


class TestSerialization:

    def _cls(self):
        return _node(type="s_node", label="S Node",
                     param_schema=[{"key": "a", "type": "number", "default": 1}])

    def test_to_dict_shape(self):
        node = self._cls()("n1", params={"a": 5})
        d = node.to_dict()
        assert d == {"id": "n1", "type": "s_node", "label": "S Node",
                     "params": {"a": 5}, "visible": True, "compute_backend": "cpu"}

    def test_note_is_omitted_when_empty_and_included_when_set(self):
        cls = self._cls()
        assert "note" not in cls("n1").to_dict()
        node = cls("n1")
        node.note = "why this node"
        assert node.to_dict()["note"] == "why this node"

    def test_round_trip_preserves_params_visible_backend_and_note(self):
        cls = self._cls()
        original = cls("n1", params={"a": 7})
        original.visible = False
        original.compute_backend = ComputeBackend.GPU
        original.note = "keep me"

        restored = cls.from_dict(original.to_dict())
        assert restored.node_id == "n1"
        assert restored.params.get("a") == 7
        assert restored.visible is False
        assert restored.compute_backend is ComputeBackend.GPU
        assert restored.note == "keep me"

    def test_from_dict_tolerates_a_bad_compute_backend(self):
        # A workflow saved by a newer/other build must not fail to load over an
        # unrecognised backend string.
        node = self._cls().from_dict({"id": "n1", "compute_backend": "quantum"})
        assert node.compute_backend is ComputeBackend.CPU

    def test_from_dict_tolerates_missing_optional_fields(self):
        node = self._cls().from_dict({"id": "n1"})
        assert node.visible is True and node.note == ""
        assert node.params.get("a") == 1

    def test_from_dict_ignores_a_non_string_note(self):
        node = self._cls().from_dict({"id": "n1", "note": 42})
        assert node.note == ""

    def test_to_dict_stores_raw_template_text_not_a_resolved_value(self):
        # Regression guard for the one-character-looking change that would
        # silently corrupt every saved workflow: going through
        # ExprParams.__getitem__ here would serialize the resolved value.
        cls = _node(type="s_raw", param_schema=[{"key": "p", "type": "text", "default": ""}])
        node = cls("n1", params={"p": "{{ $json.x }}"})
        token = set_current_data({"x": "resolved"})
        try:
            assert node.to_dict()["params"]["p"] == "{{ $json.x }}"
        finally:
            reset_current_data(token)

    def test_repr_names_the_class_and_id(self):
        assert "n1" in repr(self._cls()("n1"))


class TestRegistry:

    def test_register_adds_the_class_under_its_type(self):
        with isolated_registry():
            cls = _node(type="r_one")
            assert register(cls) is cls          # usable as a decorator
            assert NODE_REGISTRY["r_one"] is cls

    def test_register_overwrites_on_a_repeated_type(self):
        # register() itself has no collision protection — the host app's loader
        # is what does `if type not in NODE_REGISTRY`. Documented, and asserted
        # here so nobody "fixes" it and changes the app's behaviour.
        with isolated_registry():
            first = _node(type="r_dup", label="first")
            second = _node(type="r_dup", label="second")
            register(first)
            register(second)
            assert NODE_REGISTRY["r_dup"] is second

    def test_registry_is_restored_by_the_isolation_helper(self):
        before = dict(NODE_REGISTRY)
        with isolated_registry():
            register(_node(type="r_temp"))
        assert NODE_REGISTRY == before


class TestExecuteContract:

    def test_base_execute_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _node(type="e_abstract")("n1").execute(np.zeros((2, 2, 3), np.uint8))

    def test_execute_gpu_falls_back_to_execute(self):
        class _Gpu(BaseNode):
            type = "e_gpu"

            def execute(self, image, mask=None, data=None, context=None):
                return {"image": image, "mask": mask, "called": "cpu"}

        img = np.zeros((2, 2, 3), np.uint8)
        assert _Gpu("n1").execute_gpu(img)["called"] == "cpu"

    def test_compute_backend_defaults_to_cpu(self):
        assert _node(type="e_backend")("n1").compute_backend is ComputeBackend.CPU


class TestToRgb:

    def test_none_passes_through(self):
        assert to_rgb(None) is None

    def test_already_rgb_is_returned_without_a_copy(self):
        img = np.zeros((3, 4, 3), np.uint8)
        assert to_rgb(img) is img

    def test_grayscale_2d_is_broadcast(self):
        out = to_rgb(np.full((3, 4), 7, np.uint8))
        assert out.shape == (3, 4, 3)
        assert (out == 7).all()

    def test_grayscale_3d_single_channel_is_broadcast(self):
        out = to_rgb(np.full((3, 4, 1), 7, np.uint8))
        assert out.shape == (3, 4, 3)
        assert (out == 7).all()

    def test_rgba_drops_alpha(self):
        img = np.dstack([np.full((2, 2), 1, np.uint8), np.full((2, 2), 2, np.uint8),
                         np.full((2, 2), 3, np.uint8), np.full((2, 2), 200, np.uint8)])
        out = to_rgb(img)
        assert out.shape == (2, 2, 3)
        assert out[0, 0].tolist() == [1, 2, 3]

    @pytest.mark.parametrize("dtype", [np.float32, np.float16, np.float64, np.int32])
    def test_non_uint8_is_clipped_and_cast(self, dtype):
        img = np.array([[[-20, 100, 300]]], dtype=dtype)
        out = to_rgb(img)
        assert out.dtype == np.uint8
        assert out[0, 0].tolist() == [0, 100, 255]

    def test_output_is_always_uint8_hw3(self):
        for img in (np.zeros((2, 2), np.uint8), np.zeros((2, 2, 1), np.uint8),
                    np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 4), np.uint8),
                    np.zeros((2, 2, 3), np.float32)):
            out = to_rgb(img)
            assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3


class TestMaskBlend:

    def test_none_mask_returns_result_unchanged(self):
        result = np.full((2, 2, 3), 200, np.uint8)
        assert mask_blend(np.zeros((2, 2, 3), np.uint8), result, None) is result

    def test_bool_mask_selects_per_pixel(self):
        orig = np.zeros((1, 2, 3), np.uint8)
        result = np.full((1, 2, 3), 255, np.uint8)
        mask = np.array([[True, False]])
        out = mask_blend(orig, result, mask)
        assert out[0, 0].tolist() == [255, 255, 255]
        assert out[0, 1].tolist() == [0, 0, 0]

    def test_float_mask_blends_proportionally(self):
        orig = np.zeros((1, 1, 3), np.uint8)
        result = np.full((1, 1, 3), 200, np.uint8)
        out = mask_blend(orig, result, np.array([[0.5]], dtype=np.float32))
        assert out[0, 0].tolist() == [100, 100, 100]

    def test_out_of_range_mask_is_clipped(self):
        orig = np.zeros((1, 2, 3), np.uint8)
        result = np.full((1, 2, 3), 100, np.uint8)
        out = mask_blend(orig, result, np.array([[5.0, -3.0]], dtype=np.float32))
        assert out[0, 0].tolist() == [100, 100, 100]   # clipped to 1.0
        assert out[0, 1].tolist() == [0, 0, 0]         # clipped to 0.0

    def test_output_is_uint8(self):
        out = mask_blend(np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 250, np.uint8),
                         np.full((2, 2), 0.7, np.float32))
        assert out.dtype == np.uint8


class TestMaskBlendShapeGuard:
    """A mask that doesn't cover the result is a node bug, reported as one.

    Added in 1.1.0. Imagira's fuzzer found ~20 nodes dying on this with numpy's
    `operands could not be broadcast together with shapes (96,96,1) (50,50,3)`,
    which names neither the node nor which side is wrong. Same exception type as
    before (`ValueError`), better message — additive, not breaking.
    """

    def test_mismatched_mask_raises_naming_both_shapes(self):
        orig = np.zeros((32, 40, 3), np.uint8)
        result = np.zeros((32, 40, 3), np.uint8)
        mask = np.ones((64, 80), bool)
        with pytest.raises(ValueError) as exc:
            mask_blend(orig, result, mask)
        message = str(exc.value)
        assert "40x32" in message and "80x64" in message
        assert "mask" in message.lower()

    def test_matching_shapes_are_unaffected(self):
        orig = np.zeros((4, 4, 3), np.uint8)
        result = np.full((4, 4, 3), 255, np.uint8)
        mask = np.zeros((4, 4), bool)
        mask[:2] = True
        out = mask_blend(orig, result, mask)
        assert out[0, 0].tolist() == [255, 255, 255]
        assert out[3, 3].tolist() == [0, 0, 0]

    def test_none_mask_still_short_circuits(self):
        result = np.zeros((4, 4, 3), np.uint8)
        assert mask_blend(np.zeros((2, 2, 3), np.uint8), result, None) is result


class TestAsBoolMask:
    """New in 1.1.0. Only a bool mask can be used as a selection.

    Every dtype in the mask contract reaches nodes, but `image[mask]` with a
    uint8 mask is integer indexing — it returns the wrong pixels or raises
    `index 255 is out of bounds for axis 0 with size 96`.
    """

    def test_none_passes_through(self):
        assert as_bool_mask(None) is None

    def test_bool_is_returned_unchanged(self):
        mask = np.zeros((4, 4), bool)
        assert as_bool_mask(mask) is mask

    def test_uint8_zero_one(self):
        mask = np.array([[0, 1], [1, 0]], np.uint8)
        assert as_bool_mask(mask).tolist() == [[False, True], [True, False]]

    def test_uint8_zero_255(self):
        mask = np.array([[0, 255], [255, 0]], np.uint8)
        assert as_bool_mask(mask).tolist() == [[False, True], [True, False]]

    def test_float_matte_thresholds_at_half(self):
        mask = np.array([[0.0, 0.4, 0.5, 0.6, 1.0]], np.float32)
        assert as_bool_mask(mask).tolist() == [[False, False, False, True, True]]

    def test_result_is_usable_as_a_selection(self):
        """The actual point: indexing with the converted mask selects pixels."""
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        mask = np.zeros((4, 4), np.uint8)
        mask[1:3, 1:3] = 255
        selected = image[as_bool_mask(mask)]
        assert selected.shape == (4, 3)
