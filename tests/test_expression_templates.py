"""n8n-style `{{ $json.path }}` templates in node params.

Ported from the main repo's tests/test_expression_templates.py, minus the
sections that drive a real Imagira pipeline (those stay in the app's suite —
the SDK has no pipeline). The behaviours kept here are the ones a third-party
node depends on: whole-field vs. embedded resolution, cross-node param refs,
raw storage staying raw for serialization, and thread isolation.
"""

from __future__ import annotations

import threading

import pytest

from imagira_node_sdk import (
    BaseNode,
    ExpressionError,
    ExprParams,
    reset_current_data,
    resolve_template,
    set_current_data,
)


class TestResolveTemplate:

    def test_non_string_passes_through_unchanged(self):
        assert resolve_template(42, {"x": 1}) == 42
        assert resolve_template(None, {"x": 1}) is None
        assert resolve_template([1, 2], {"x": 1}) == [1, 2]

    def test_string_without_braces_passes_through_unchanged(self):
        assert resolve_template("plain text", {"x": 1}) == "plain text"

    def test_whole_string_expression_preserves_native_type(self):
        data = {"fps": 30, "ok": True, "nested": {"a": 1}}
        assert resolve_template("{{ $json.fps }}", data) == 30
        assert resolve_template("{{ $json.fps }}", data) != "30"
        assert resolve_template("{{ $json.ok }}", data) is True
        assert resolve_template("{{ $json.nested }}", data) == {"a": 1}

    def test_data_alias_and_json_alias_are_equivalent(self):
        data = {"fps": 30}
        assert resolve_template("{{ $data.fps }}", data) == resolve_template("{{ $json.fps }}", data)

    def test_dot_path_into_nested_object(self):
        data = {"article": {"title": "Hello", "meta": {"id": 7}}}
        assert resolve_template("{{ $json.article.title }}", data) == "Hello"
        assert resolve_template("{{ $json.article.meta.id }}", data) == 7

    def test_bracket_index_into_list(self):
        data = {"items": [{"name": "first"}, {"name": "second"}]}
        assert resolve_template("{{ $json.items[0].name }}", data) == "first"
        assert resolve_template("{{ $json.items[1].name }}", data) == "second"

    def test_embedded_expression_stringifies_in_place(self):
        data = {"name": "frame_007", "fps": 24}
        out = resolve_template("{{ $json.name }}_{{ $json.fps }}fps.png", data)
        assert out == "frame_007_24fps.png"

    def test_embedded_dict_value_serializes_as_json(self):
        data = {"article": {"title": "Hello", "id": 7}}
        out = resolve_template("meta: {{ $json.article }}", data)
        assert out.startswith("meta: {")
        assert '"title": "Hello"' in out
        assert '"id": 7' in out

    def test_missing_key_raises_expression_error(self):
        with pytest.raises(ExpressionError):
            resolve_template("{{ $json.does_not_exist }}", {"a": 1})

    def test_no_data_at_all_raises_on_path_access(self):
        with pytest.raises(ExpressionError):
            resolve_template("{{ $json.anything }}", None)

    def test_no_data_at_all_is_fine_when_whole_json_referenced(self):
        assert resolve_template("{{ $json }}", None) is None


class TestResolveTemplateCrossNodeParams:
    """`$nodes["other_id"].key` — reach another node's own configured param
    value directly, regardless of what (if anything) connects the two nodes.
    """

    def test_reads_another_nodes_param_by_id(self):
        all_nodes = {"chroma_key_1": {"tolerance": 42, "key_color": "#00ff00"}}
        assert resolve_template("{{ $nodes['chroma_key_1'].tolerance }}", all_nodes=all_nodes) == 42

    def test_whole_field_reference_preserves_native_type(self):
        all_nodes = {"n1": {"strength": 85}}
        out = resolve_template("{{ $nodes['n1'].strength }}", all_nodes=all_nodes)
        assert out == 85 and isinstance(out, int)

    def test_embedded_reference_stringifies(self):
        all_nodes = {"n1": {"strength": 85}}
        out = resolve_template("strength={{ $nodes['n1'].strength }}%", all_nodes=all_nodes)
        assert out == "strength=85%"

    def test_raw_upstream_expression_is_not_recursively_resolved(self):
        # If the referenced node's OWN param is itself a template, the reader
        # gets that literal text back rather than a further-resolved value —
        # deliberate, see resolve_template's docstring (avoids A<->B cycles
        # and evaluating another node's expression against the wrong data).
        all_nodes = {"n1": {"strength": "{{ $json.something }}"}}
        out = resolve_template("{{ $nodes['n1'].strength }}", all_nodes=all_nodes)
        assert out == "{{ $json.something }}"

    def test_missing_node_id_raises_expression_error(self):
        with pytest.raises(ExpressionError):
            resolve_template("{{ $nodes['does_not_exist'].tolerance }}", all_nodes={"n1": {}})

    def test_missing_param_key_raises_expression_error(self):
        with pytest.raises(ExpressionError):
            resolve_template("{{ $nodes['n1'].nope }}", all_nodes={"n1": {"tolerance": 1}})

    def test_no_all_nodes_at_all_raises_a_clear_error_not_a_crash(self):
        with pytest.raises(ExpressionError):
            resolve_template("{{ $nodes['n1'].tolerance }}")

    def test_data_and_nodes_can_be_combined_in_one_expression(self):
        data = {"suffix": "px"}
        all_nodes = {"n1": {"tolerance": 30}}
        out = resolve_template("{{ $nodes['n1'].tolerance }}{{ $json.suffix }}", data, all_nodes)
        assert out == "30px"

    def test_two_adjacent_whole_expressions_with_no_separator_both_resolve(self):
        # Regression: the "whole field" check used to be a single greedy
        # ^\{\{(.*)\}\}$ pattern, which on "{{ a }}{{ b }}" matched from the
        # FIRST `{{` to the LAST `}}`, treating the entire string (including
        # the middle `}}{{`) as one expression — a syntax error.
        all_nodes = {"n1": {"tolerance": 30, "strength": 7}}
        out = resolve_template("{{ $nodes['n1'].tolerance }}{{ $nodes['n1'].strength }}",
                               all_nodes=all_nodes)
        assert out == "307"


class TestExprParams:

    def test_get_resolves_template_against_current_data(self):
        p = ExprParams({"name_pattern": "{{ $json.name }}.png"})
        token = set_current_data({"name": "frame_003"})
        try:
            assert p.get("name_pattern") == "frame_003.png"
        finally:
            reset_current_data(token)

    def test_getitem_also_resolves(self):
        p = ExprParams({"name_pattern": "{{ $json.name }}"})
        token = set_current_data({"name": "x"})
        try:
            assert p["name_pattern"] == "x"
        finally:
            reset_current_data(token)

    def test_plain_values_unaffected_with_no_current_data(self):
        p = ExprParams({"fps": 24, "path": "plain/path.mp4"})
        assert p.get("fps") == 24
        assert p.get("path") == "plain/path.mp4"

    def test_raw_storage_is_reachable_for_serialization(self):
        # dict.items()/dict.get() bypass the override entirely — this is what
        # BaseNode.to_dict() relies on to save the raw `{{ }}` text, not
        # whatever it last resolved to, into a workflow file.
        p = ExprParams({"name_pattern": "{{ $json.name }}.png"})
        token = set_current_data({"name": "resolved_value"})
        try:
            assert dict(dict.items(p))["name_pattern"] == "{{ $json.name }}.png"
        finally:
            reset_current_data(token)

    def test_concurrent_threads_do_not_see_each_others_current_data(self):
        p = ExprParams({"v": "{{ $json.who }}"})
        results = {}

        def worker(name):
            token = set_current_data({"who": name})
            try:
                for _ in range(200):
                    assert p.get("v") == name
                results[name] = p.get("v")
            finally:
                reset_current_data(token)

        threads = [threading.Thread(target=worker, args=(n,)) for n in ("alice", "bob", "carol")]
        for t in threads: t.start()
        for t in threads: t.join()
        assert results == {"alice": "alice", "bob": "bob", "carol": "carol"}


class TestBaseNodeSerializationStaysRaw:

    def test_to_dict_saves_the_template_text_not_a_resolved_value(self):
        class _ExprTestNode(BaseNode):
            type = "expr_test_node_unregistered"
            param_schema = [
                {"key": "pattern", "label": "Pattern", "type": "text", "default": ""},
            ]

            def execute(self, image=None, mask=None, data=None, context=None):
                return {"image": image, "mask": mask}

        node = _ExprTestNode("n1", params={"pattern": "{{ $json.name }}.png"})
        token = set_current_data({"name": "should_not_leak_into_to_dict"})
        try:
            d = node.to_dict()
        finally:
            reset_current_data(token)
        assert d["params"]["pattern"] == "{{ $json.name }}.png"


class TestSingleContextVar:
    """The hazard the SDK exists to prevent, asserted directly.

    ``ExprParams`` resolves against a module-level ContextVar that the *host*
    binds via ``set_current_data``. If a second copy of this module ever ends up
    in the process (a vendored fork, a stale shim), the setter and the reader
    stop meeting and every node behaves as if no data were bound: a whole-field
    ``{{ $json }}`` silently yields None, and a path access raises an
    ExpressionError that reads like the user typed a bad expression. These tests
    fail loudly if that ever happens.
    """

    def test_setter_and_reader_share_one_context_var(self):
        from imagira_node_sdk import expression as via_package
        from imagira_node_sdk.base_node import ExprParams as via_base_node

        assert via_base_node is ExprParams
        assert via_package.set_current_data is set_current_data
        assert via_package._CURRENT_CTX is via_package._CURRENT_CTX

    def test_binding_via_public_api_reaches_a_nodes_params(self):
        class _Node(BaseNode):
            type = "ctxvar_probe_node"
            param_schema = [{"key": "p", "label": "P", "type": "text", "default": ""}]

            def execute(self, image=None, mask=None, data=None, context=None):
                return {"image": image, "value": self.params.get("p")}

        node = _Node("n1", params={"p": "{{ $json.v }}"})
        token = set_current_data({"v": "resolved"})
        try:
            assert node.params.get("p") == "resolved"
        finally:
            reset_current_data(token)
