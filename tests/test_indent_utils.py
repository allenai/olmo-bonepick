"""Tests for indentation utilities."""

import pytest

from bonepick.data.indentation import (
    convert_spaces_to_tabs,
    detect_indentation,
    has_space_indentation,
)


class TestHasSpaceIndentation:
    """Tests for the fast heuristic check."""

    def test_no_indentation(self):
        text = "line1\nline2\nline3"
        assert has_space_indentation(text) is False

    def test_tab_indentation(self):
        text = "def foo():\n\tprint('hello')\n\tprint('world')"
        assert has_space_indentation(text) is False

    def test_space_indentation_2(self):
        text = "def foo():\n  print('hello')"
        assert has_space_indentation(text) is True

    def test_space_indentation_4(self):
        text = "def foo():\n    print('hello')"
        assert has_space_indentation(text) is True

    def test_first_line_indented(self):
        text = "  indented first line\nnormal line"
        assert has_space_indentation(text) is True

    def test_empty_text(self):
        assert has_space_indentation("") is False

    def test_single_space_not_detected(self):
        # Single space after newline is not indentation
        text = "word\n word"  # just one space
        assert has_space_indentation(text) is False

    def test_indentation_beyond_sample_size(self):
        # Indentation after sample_size should still work with small sample
        text = "a" * 100 + "\n  indented"
        assert has_space_indentation(text, sample_size=50) is False
        assert has_space_indentation(text, sample_size=200) is True

    def test_mixed_content_with_spaces(self):
        text = "some text\nmore text\n    indented line\nback to normal"
        assert has_space_indentation(text) is True


class TestDetectIndentation:
    """Tests for the main GCD-based detection."""

    def test_2_space_indent(self):
        text = """def foo():
  if True:
    print('hello')
  return"""
        assert detect_indentation(text) == 2

    def test_4_space_indent(self):
        text = """def foo():
    if True:
        print('hello')
    return"""
        assert detect_indentation(text) == 4

    def test_8_space_indent(self):
        text = """def foo():
        if True:
                print('hello')
        return"""
        assert detect_indentation(text) == 8

    def test_no_indentation(self):
        text = "line1\nline2\nline3"
        assert detect_indentation(text) is None

    def test_empty_text(self):
        assert detect_indentation("") is None

    def test_only_blank_lines(self):
        text = "\n\n\n"
        assert detect_indentation(text) is None

    def test_tab_indentation_ignored(self):
        text = "def foo():\n\tprint('hello')"
        assert detect_indentation(text) is None

    def test_mixed_tabs_spaces_ignores_tabs(self):
        text = "def foo():\n\ttab line\n    space line\n        more space"
        assert detect_indentation(text) == 4

    def test_irregular_indentation_gcd(self):
        # Lines indented 3 and 6 spaces -> GCD is 3
        text = "base\n   three\n      six"
        assert detect_indentation(text) == 3

    def test_single_indented_line(self):
        text = "def foo():\n    return True"
        assert detect_indentation(text) == 4

    def test_deeply_nested(self):
        text = """class Foo:
    def bar(self):
        if True:
            for i in range(10):
                if i > 5:
                    print(i)"""
        assert detect_indentation(text) == 4

    def test_early_exit_on_gcd_1(self):
        # If we find indent of 3 then 4, GCD becomes 1
        text = "a\n   three spaces\n    four spaces"
        assert detect_indentation(text) == 1


class TestRealWorldExamples:
    """Tests with realistic code snippets."""

    def test_python_class(self):
        text = '''class MyClass:
    """A sample class."""

    def __init__(self, value):
        self.value = value

    def process(self):
        if self.value > 0:
            return self.value * 2
        else:
            return 0
'''
        assert detect_indentation(text) == 4
        assert has_space_indentation(text) is True

    def test_javascript_2_space(self):
        text = """function foo() {
  if (true) {
    console.log("hello");
  }
  return 42;
}
"""
        assert detect_indentation(text) == 2
        assert has_space_indentation(text) is True

    def test_yaml_2_space(self):
        text = """root:
  child1:
    nested: value
  child2:
    - item1
    - item2
"""
        assert detect_indentation(text) == 2

    def test_json_like(self):
        text = """{
    "key": {
        "nested": "value"
    }
}
"""
        assert detect_indentation(text) == 4


class TestConvertSpacesToTabs:
    """Tests for space-to-tab conversion."""

    def test_4_space_to_tabs(self):
        text = "def foo():\n    return True"
        expected = "def foo():\n\treturn True"
        assert convert_spaces_to_tabs(text) == expected

    def test_2_space_to_tabs(self):
        text = "def foo():\n  if True:\n    return"
        expected = "def foo():\n\tif True:\n\t\treturn"
        assert convert_spaces_to_tabs(text) == expected

    def test_nested_indentation(self):
        text = "a\n    b\n        c\n            d"
        expected = "a\n\tb\n\t\tc\n\t\t\td"
        assert convert_spaces_to_tabs(text) == expected

    def test_explicit_indent_size(self):
        text = "a\n  b\n    c"
        # Force 4-space interpretation (so 2 spaces = 0 tabs + 2 remainder)
        expected = "a\n  b\n\tc"
        assert convert_spaces_to_tabs(text, indent_size=4) == expected

    def test_no_indentation_unchanged(self):
        text = "line1\nline2\nline3"
        assert convert_spaces_to_tabs(text) == text

    def test_tab_indented_unchanged(self):
        text = "def foo():\n\treturn True"
        assert convert_spaces_to_tabs(text) == text

    def test_preserves_inline_spaces(self):
        text = "def foo():\n    return a + b"
        expected = "def foo():\n\treturn a + b"
        assert convert_spaces_to_tabs(text) == expected

    def test_remainder_spaces_preserved(self):
        # With indent_size=2: 3 spaces = 1 tab + 1 space
        text = "a\n  b\n   c"
        expected = "a\n\tb\n\t c"
        assert convert_spaces_to_tabs(text, indent_size=2) == expected

    def test_empty_lines_preserved(self):
        text = "def foo():\n\n    return True\n"
        expected = "def foo():\n\n\treturn True\n"
        assert convert_spaces_to_tabs(text) == expected

    def test_mixed_content_only_leading_converted(self):
        text = "    code with    multiple   spaces inside"
        expected = "\tcode with    multiple   spaces inside"
        assert convert_spaces_to_tabs(text, indent_size=4) == expected
