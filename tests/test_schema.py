"""param_schema validation — mostly about catching the silent-fallback bugs."""

from __future__ import annotations

from imagira_node_sdk import BaseNode, validate_param_schema
from imagira_node_sdk.schema import ERROR, WARNING


def levels(issues):
    return [i.level for i in issues]


def messages(issues):
    return " | ".join(str(i) for i in issues)


class TestCleanSchemas:

    def test_a_fully_specified_schema_has_no_issues(self):
        assert validate_param_schema([
            {"key": "amount", "label": "Amount", "type": "slider",
             "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "help": "How much"},
            {"key": "mode", "label": "Mode", "type": "select",
             "default": "soft", "options": ["soft", "hard"]},
            {"key": "enabled", "label": "Enabled", "type": "bool", "default": True},
            {"key": "tint", "label": "Tint", "type": "colour", "default": "#00ff00"},
            {"key": "name", "label": "Name", "type": "text", "default": "",
             "placeholder": "out.png", "browse": "folder"},
            {"key": "hard_only", "label": "Hard only", "type": "number", "default": 1,
             "show_if": {"key": "mode", "values": ["hard"]}},
        ]) == []

    def test_an_empty_schema_is_valid(self):
        assert validate_param_schema([]) == []

    def test_accepts_a_node_class_directly(self):
        class _N(BaseNode):
            type = "schema_probe"
            param_schema = [{"key": "a", "label": "A", "type": "text", "default": ""}]

        assert validate_param_schema(_N) == []


class TestFatalProblems:

    def test_non_list_schema(self):
        issues = validate_param_schema({"key": "a"})
        assert levels(issues) == [ERROR] and "must be a list" in messages(issues)

    def test_non_dict_entry(self):
        assert any(i.level == ERROR for i in validate_param_schema(["nope"]))

    def test_missing_key_is_flagged_as_the_keyerror_it_causes(self):
        issues = validate_param_schema([{"label": "A", "type": "text"}])
        assert any(i.level == ERROR and "KeyError" in i.message for i in issues)

    def test_duplicate_keys(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "text", "default": ""},
            {"key": "a", "label": "B", "type": "text", "default": ""},
        ])
        assert any(i.level == ERROR and "duplicate" in i.message for i in issues)

    def test_missing_type(self):
        issues = validate_param_schema([{"key": "a", "label": "A", "default": ""}])
        assert any(i.level == ERROR and "missing 'type'" in i.message for i in issues)

    def test_min_above_max(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "slider", "default": 1, "min": 10, "max": 1}])
        assert any(i.level == ERROR and "greater than max" in i.message for i in issues)

    def test_non_positive_step(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "slider", "default": 1,
             "min": 0, "max": 2, "step": 0}])
        assert any(i.level == ERROR and "positive" in i.message for i in issues)

    def test_select_without_options(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "select", "default": "x"}])
        assert any(i.level == ERROR and "requires a non-empty 'options'" in i.message
                   for i in issues)

    def test_select_options_as_objects_rather_than_strings(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "select", "default": "x",
             "options": [{"value": "x", "label": "X"}]}])
        assert any(i.level == ERROR and "plain strings" in i.message for i in issues)

    def test_show_if_pointing_at_a_nonexistent_param(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "text", "default": "",
             "show_if": {"key": "ghost", "values": ["x"]}}])
        assert any(i.level == ERROR and "never be shown" in i.message for i in issues)

    def test_show_if_pointing_at_itself(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "text", "default": "",
             "show_if": {"key": "a", "values": ["x"]}}])
        assert any(i.level == ERROR and "itself" in i.message for i in issues)

    def test_show_if_malformed(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "text", "default": "", "show_if": "mode==hard"}])
        assert any(i.level == ERROR for i in issues)

    def test_bad_browse_kind(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "text", "default": "", "browse": "file"}])
        assert any(i.level == ERROR and "browse" in i.message for i in issues)


class TestSilentFallbackWarnings:
    """The class of bug this validator exists for: it loads, it just doesn't do
    what the author meant, and nothing anywhere says so."""

    def test_misspelled_colour_suggests_the_canonical_spelling(self):
        issues = validate_param_schema([
            {"key": "c", "label": "C", "type": "color", "default": "#ffffff"}])
        msg = messages(issues)
        assert "did you mean 'colour'" in msg
        assert "text input" in msg

    def test_int_suggests_number(self):
        issues = validate_param_schema([{"key": "n", "label": "N", "type": "int", "default": 1}])
        assert "did you mean 'number'" in messages(issues)

    def test_string_suggests_text(self):
        issues = validate_param_schema([{"key": "s", "label": "S", "type": "string",
                                         "default": ""}])
        assert "did you mean 'text'" in messages(issues)

    def test_wholly_unknown_type_still_warns_without_a_suggestion(self):
        issues = validate_param_schema([{"key": "x", "label": "X", "type": "spinner",
                                         "default": 1}])
        assert any(i.level == WARNING and "unknown type" in i.message for i in issues)

    def test_missing_default_explains_the_none(self):
        issues = validate_param_schema([{"key": "a", "label": "A", "type": "text"}])
        assert any("None" in i.message for i in issues)

    def test_missing_label(self):
        issues = validate_param_schema([{"key": "a", "type": "text", "default": ""}])
        assert any("label" in i.message for i in issues)

    def test_slider_without_a_range(self):
        issues = validate_param_schema([{"key": "a", "label": "A", "type": "slider",
                                         "default": 1}])
        assert any("no range to draw" in i.message for i in issues)

    def test_default_outside_the_declared_range(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "slider", "default": 5, "min": 0, "max": 1}])
        assert any("above max" in i.message for i in issues)

    def test_select_default_not_among_the_options(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "select", "default": "z",
             "options": ["x", "y"]}])
        assert any("not one of 'options'" in i.message for i in issues)

    def test_bool_default_that_is_not_a_bool(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "bool", "default": "true"}])
        assert any("True/False" in i.message for i in issues)

    def test_colour_default_that_is_not_a_hex_string(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "colour", "default": "green"}])
        assert any("#rrggbb" in i.message for i in issues)

    def test_options_on_a_non_select_param(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "text", "default": "", "options": ["x"]}])
        assert any("only meaningful for type 'select'" in i.message for i in issues)

    def test_browse_on_a_non_text_param(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "number", "default": 1, "browse": "folder"}])
        assert any("only applies to a 'text' param" in i.message for i in issues)

    def test_typo_in_a_key_name_is_caught(self):
        issues = validate_param_schema([
            {"key": "a", "label": "A", "type": "text", "default": "", "helptext": "oops"}])
        assert any("unrecognised key 'helptext'" in i.message for i in issues)


class TestIssueFormatting:

    def test_str_includes_level_location_and_message(self):
        issue = validate_param_schema([{"key": "a", "type": "text", "default": ""}])[0]
        s = str(issue)
        assert s.startswith("[warning]") and "'a'" in s

    def test_issues_are_hashable_and_comparable(self):
        a = validate_param_schema([{"key": "a", "type": "text", "default": ""}])
        b = validate_param_schema([{"key": "a", "type": "text", "default": ""}])
        assert a == b and len(set(a)) == len(a)
