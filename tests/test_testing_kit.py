"""The author-facing test kit — tested itself, since node authors trust it to
tell them the truth about their node."""

from __future__ import annotations

import numpy as np
import pytest

from imagira_node_sdk import NODE_REGISTRY, BaseNode, register, to_rgb
from imagira_node_sdk.testing import (
    assert_node_contract,
    isolated_registry,
    make_image,
    make_mask,
    make_samples,
    run_node,
)


class GoodNode(BaseNode):
    """A node that does everything right — the reference for these tests."""

    type = "testkit_invert"
    label = "Invert"
    category = "Post-Process"
    description = "Inverts every channel."
    param_schema = [
        {"key": "amount", "label": "Amount", "type": "slider",
         "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
        {"key": "tag", "label": "Tag", "type": "text", "default": ""},
    ]

    def execute(self, image, mask=None, data=None, context=None):
        amount = float(self.params.get("amount", 1.0))
        src = to_rgb(image).astype(np.float32)
        out = (255.0 - src) * amount + src * (1.0 - amount)
        return {"image": np.clip(out, 0, 255).astype(np.uint8), "mask": mask}


class TestFixtureHelpers:

    def test_make_image_shape_and_dtype(self):
        img = make_image(3, 5, (10, 20, 30))
        assert img.shape == (3, 5, 3) and img.dtype == np.uint8
        assert img[0, 0].tolist() == [10, 20, 30]

    def test_make_mask_shape_and_dtype(self):
        assert make_mask(3, 4).shape == (3, 4)
        assert make_mask(3, 4).dtype == np.bool_
        assert not make_mask(2, 2, filled=False).any()

    def test_make_samples_matches_the_documented_shape(self):
        img = make_image(2, 2, (10, 20, 30))
        samples = make_samples(img)
        assert samples["image_size"] == (2, 2)
        island = samples["islands"][0]
        assert island["pixels"].shape == (4, 3)
        assert island["positions"].shape == (4, 2)
        assert island["area"] == 4
        assert island["mean_color"] == [10, 20, 30]
        assert island["label"] == 1

    def test_make_samples_honours_a_partial_mask(self):
        img = make_image(2, 2)
        mask = np.array([[True, False], [False, False]])
        island = make_samples(img, mask)["islands"][0]
        assert island["area"] == 1

    def test_isolated_registry_restores_on_exception(self):
        before = dict(NODE_REGISTRY)
        with pytest.raises(RuntimeError):
            with isolated_registry():
                register(type("Tmp", (BaseNode,), {"type": "leaky_node"}))
                assert "leaky_node" in NODE_REGISTRY
                raise RuntimeError("boom")
        assert dict(NODE_REGISTRY) == before


class TestRunNode:

    def test_runs_and_returns_the_result_dict(self):
        out = run_node(GoodNode, image=make_image(2, 2, (10, 20, 30)))
        assert out["image"][0, 0].tolist() == [245, 235, 225]

    def test_supplies_a_default_image_when_none_given(self):
        assert run_node(GoodNode)["image"].shape == (8, 8, 3)

    def test_params_are_applied(self):
        out = run_node(GoodNode, image=make_image(1, 1, (10, 10, 10)), params={"amount": 0.0})
        assert out["image"][0, 0].tolist() == [10, 10, 10]

    def test_binds_the_data_context_so_templates_resolve(self):
        # Without set_current_data around execute(), the node would see the raw
        # "{{ ... }}" text and the test would pass for the wrong reason.
        class _TagNode(BaseNode):
            type = "testkit_tag"

            def execute(self, image, mask=None, data=None, context=None):
                return {"image": image, "data": {"tag": self.params.get("tag")}}

            param_schema = [{"key": "tag", "label": "Tag", "type": "text", "default": ""}]

        out = run_node(_TagNode, params={"tag": "{{ $json.name }}.png"}, data={"name": "frame_1"})
        assert out["data"]["tag"] == "frame_1.png"

    def test_binds_all_nodes_for_cross_node_param_refs(self):
        class _RefNode(BaseNode):
            type = "testkit_ref"
            param_schema = [{"key": "strength", "label": "S", "type": "number", "default": 0}]

            def execute(self, image, mask=None, data=None, context=None):
                return {"image": image, "data": {"s": self.params.get("strength")}}

        out = run_node(_RefNode, params={"strength": "{{ $nodes['ck'].tolerance }}"},
                       all_nodes={"ck": {"tolerance": 63}})
        assert out["data"]["s"] == 63

    def test_context_is_forwarded(self):
        class _CtxNode(BaseNode):
            type = "testkit_ctx"

            def execute(self, image, mask=None, data=None, context=None):
                return {"image": image, "data": dict(context or {})}

        assert run_node(_CtxNode, context={"run_id": 7})["data"] == {"run_id": 7}

    def test_resets_the_data_context_even_when_execute_raises(self):
        class _BoomNode(BaseNode):
            type = "testkit_boom"

            def execute(self, image, mask=None, data=None, context=None):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            run_node(_BoomNode, data={"x": 1})
        # a leaked binding would make this resolve instead of raising
        from imagira_node_sdk import ExpressionError, resolve_template
        with pytest.raises(ExpressionError):
            resolve_template("{{ $json.x }}", None)


class TestRunNodeResultChecking:

    def _bad(self, name, result, **attrs):
        body = {"type": f"testkit_bad_{name}",
                "execute": lambda self, image, mask=None, data=None, context=None: result}
        body.update(attrs)
        return type(f"Bad{name}", (BaseNode,), body)

    def test_none_return_is_rejected_with_an_explanation(self):
        with pytest.raises(AssertionError, match="returned None"):
            run_node(self._bad("none", None))

    def test_non_dict_return_is_rejected(self):
        with pytest.raises(AssertionError, match="must return a dict"):
            run_node(self._bad("list", []))

    def test_unrecognised_result_key_is_rejected(self):
        img = make_image(2, 2)
        with pytest.raises(AssertionError, match="unrecognised key"):
            run_node(self._bad("typo", {"images": img}), image=img)

    def test_float_image_is_rejected_with_the_fix_named(self):
        bad = self._bad("float", {"image": np.zeros((2, 2, 3), np.float32)})
        with pytest.raises(AssertionError, match="must be uint8"):
            run_node(bad, image=make_image(2, 2))

    def test_grayscale_image_is_rejected(self):
        bad = self._bad("gray", {"image": np.zeros((2, 2), np.uint8)})
        with pytest.raises(AssertionError, match=r"\(H, W, 3\)"):
            run_node(bad, image=make_image(2, 2))

    def test_uint8_mask_is_rejected(self):
        img = make_image(2, 2)
        bad = self._bad("mask255", {"image": img, "mask": np.full((2, 2), 255, np.uint8)})
        with pytest.raises(AssertionError, match="must be np.bool_"):
            run_node(bad, image=img)

    def test_mask_size_mismatch_is_rejected(self):
        img = make_image(4, 4)
        bad = self._bad("mismatch", {"image": img, "mask": make_mask(2, 2)})
        with pytest.raises(AssertionError, match="does not match image"):
            run_node(bad, image=img)

    def test_missing_image_from_a_producing_node_is_rejected(self):
        bad = self._bad("noimg", {"image": None})
        with pytest.raises(AssertionError, match="produces_image is True"):
            run_node(bad, image=make_image(2, 2))

    def test_a_sink_node_may_return_no_image(self):
        sink = self._bad("sink", {"image": None}, has_output_port=False,
                         produces_image=False, is_sink_node=True)
        assert run_node(sink, image=make_image(2, 2)) == {"image": None}

    def test_samples_without_islands_is_rejected(self):
        img = make_image(2, 2)
        bad = self._bad("samples", {"image": img, "samples": {"image_size": (2, 2)}},
                        produces_samples=True)
        with pytest.raises(AssertionError, match="islands"):
            run_node(bad, image=img)

    def test_non_dict_data_is_rejected(self):
        img = make_image(2, 2)
        bad = self._bad("data", {"image": img, "data": [1, 2]}, produces_data=True)
        with pytest.raises(AssertionError, match='"data"'):
            run_node(bad, image=img)

    def test_check_result_can_be_switched_off(self):
        bad = self._bad("unchecked", {"image": np.zeros((2, 2, 3), np.float32)})
        run_node(bad, image=make_image(2, 2), check_result=False)


class TestAssertNodeContract:

    def test_a_good_node_passes(self):
        assert_node_contract(GoodNode)

    def test_non_node_is_rejected(self):
        with pytest.raises(AssertionError, match="not a BaseNode subclass"):
            assert_node_contract(dict)

    def test_base_node_itself_is_rejected(self):
        with pytest.raises(AssertionError, match="not a node"):
            assert_node_contract(BaseNode)

    def test_missing_type_is_reported(self):
        with pytest.raises(AssertionError, match="still 'base'"):
            assert_node_contract(type("NoType", (BaseNode,), {
                "execute": lambda self, image, mask=None, data=None, context=None: {}}))

    def test_unnamespaced_type_is_reported(self):
        with pytest.raises(AssertionError, match="not namespaced"):
            assert_node_contract(type("Bare", (BaseNode,), {
                "type": "invert", "label": "Invert",
                "execute": lambda self, image, mask=None, data=None, context=None: {}}))

    def test_uppercase_type_is_reported(self):
        with pytest.raises(AssertionError, match="permanent identifier"):
            assert_node_contract(type("Upper", (BaseNode,), {
                "type": "My_Node", "label": "X",
                "execute": lambda self, image, mask=None, data=None, context=None: {}}))

    def test_inherited_type_is_reported(self):
        parent = type("Parent", (BaseNode,), {
            "type": "author_parent", "label": "P",
            "execute": lambda self, image, mask=None, data=None, context=None: {}})
        with pytest.raises(AssertionError, match="inherited, not set on this class"):
            assert_node_contract(type("Child", (parent,), {}))

    def test_unimplemented_execute_is_reported(self):
        with pytest.raises(AssertionError, match="not overridden"):
            assert_node_contract(type("NoExec", (BaseNode,), {
                "type": "author_noexec", "label": "X"}))

    def test_accepts_flag_without_the_matching_parameter_is_reported(self):
        # The host calls execute(image, mask, samples=...) only for nodes that
        # set accepts_samples — so this combination is a TypeError on first run.
        # Since 1.2.0 this is rejected at class creation, before the kit runs.
        with pytest.raises(TypeError, match="no `samples` parameter"):
            assert_node_contract(type("NoSamplesArg", (BaseNode,), {
                "type": "author_nosamples", "label": "X", "accepts_samples": True,
                "execute": lambda self, image, mask=None, data=None, context=None: {}}))

    @pytest.mark.parametrize("flag,kwarg", [
        ("accepts_samples", "samples"), ("accepts_data", "data"),
        ("accepts_context", "context"),
    ])
    def test_every_opt_in_kwarg_is_checked(self, flag, kwarg):
        with pytest.raises(TypeError, match=f"no `{kwarg}` parameter"):
            assert_node_contract(type("Missing", (BaseNode,), {
                "type": f"author_missing_{kwarg}", "label": "X", flag: True,
                "execute": lambda self, image, mask=None: {}}))

    def test_the_canonical_four_arg_signature_is_accepted_without_any_flags(self):
        # BaseNode.execute's own shape. Declaring data/context without the
        # matching accepts_* flag is normal and must not be reported.
        assert_node_contract(type("Canonical", (BaseNode,), {
            "type": "author_canonical", "label": "X",
            "execute": lambda self, image, mask=None, data=None, context=None: {}}))

    def test_a_short_signature_is_fine_when_no_flags_opt_in(self):
        assert_node_contract(type("ShortSig", (BaseNode,), {
            "type": "author_shortsig", "label": "X",
            "execute": lambda self, image, mask=None: {}}))

    def test_execute_with_kwargs_is_accepted(self):
        assert_node_contract(type("Kwargs", (BaseNode,), {
            "type": "author_kwargs", "label": "X",
            "execute": lambda self, image, **kwargs: {}}))

    def test_empty_label_is_reported(self):
        with pytest.raises(AssertionError, match="`label` is empty"):
            assert_node_contract(type("NoLabel", (BaseNode,), {
                "type": "author_nolabel", "label": "",
                "execute": lambda self, image, mask=None, data=None, context=None: {}}))

    def test_sink_with_an_output_port_is_reported(self):
        with pytest.raises(AssertionError, match="terminal by definition"):
            assert_node_contract(type("BadSink", (BaseNode,), {
                "type": "author_badsink", "label": "X", "is_sink_node": True,
                "execute": lambda self, image, mask=None, data=None, context=None: {}}))

    def test_source_with_an_input_port_is_reported(self):
        with pytest.raises(AssertionError, match="generates its own image"):
            assert_node_contract(type("BadSource", (BaseNode,), {
                "type": "author_badsource", "label": "X", "is_source_node": True,
                "execute": lambda self, image, mask=None, data=None, context=None: {}}))

    def test_supports_gpu_without_execute_gpu_is_reported(self):
        with pytest.raises(AssertionError, match="silently run on the CPU path"):
            assert_node_contract(type("FakeGpu", (BaseNode,), {
                "type": "author_fakegpu", "label": "X", "supports_gpu": True,
                "execute": lambda self, image, mask=None, data=None, context=None: {}}))

    def test_inherited_i18n_with_content_is_reported(self):
        parent = type("I18nParent", (BaseNode,), {
            "type": "author_i18n_parent", "label": "P",
            "i18n": {"de": {"label": "P"}},
            "execute": lambda self, image, mask=None, data=None, context=None: {}})
        with pytest.raises(AssertionError, match="shared class-level dict"):
            assert_node_contract(type("I18nChild", (parent,), {"type": "author_i18n_child"}))

    def test_schema_errors_fail_but_warnings_pass_by_default(self):
        # Since 1.2.0 an unknown param type is rejected at class creation.
        with pytest.raises(TypeError, match="colour"):
            type("Warny", (BaseNode,), {
                "type": "author_warny", "label": "X",
                "param_schema": [{"key": "c", "label": "C", "type": "color", "default": "#fff"}],
                "execute": lambda self, image, mask=None, data=None, context=None: {}})

    def test_all_problems_are_reported_at_once(self):
        with pytest.raises(AssertionError) as exc:
            assert_node_contract(type("Messy", (BaseNode,), {
                "type": "Bare Type", "label": "",
                "execute": lambda self, image, mask=None: {}}))
        # not namespaced + uppercase-with-space + empty label + accepts_data
        # with no `data` parameter
        assert str(exc.value).count("  - ") >= 3
