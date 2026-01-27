import dataclasses as dt

from bonepick.annotate.prompts import BaseAnnotationPrompt, DataclassType

from .code_rubrics import BetterTruncationCodePrompt


@dt.dataclass(frozen=True)
class BaseStackEduReduxOutput:
    justification: str
    score: int


@dt.dataclass(frozen=True)
class BaseStackEduReduxPrompt(BetterTruncationCodePrompt):
    instructions: str = """
After examining the extract, respond with a JSON object with the following format:

```json
{{
    "justification": "...",    # a brief justification of the score, up to 100 words
    "score": int,              # the final score between 1 and 5 (inclusive)
}}
```
"""

    def format_text(self, text: str, max_text_length: int | None = None) -> str:
        # save 40 characters for the info about chopped text
        max_text_length = max_text_length - 80 if max_text_length is not None else None

        if max_text_length is not None and len(text) > max_text_length:
            # find the closest "\n" before the max_text_length
            closest_newline = p if (p := text.rfind("\n", 0, max_text_length)) > -1 else max_text_length
            text = text[:closest_newline]
            remaining_text = text[closest_newline:]

            remaining_chars = len(remaining_text)
            remaining_lines = remaining_text.count("\n")
            text = f"{text.strip()}\n<... truncated {remaining_chars:,} characters, {remaining_lines:,} lines ...>"

        return (
            f"\n\n=========== BEGIN OF EXTRACT ===========\n{text}\n=========== END OF EXTRACT =============\n\n"
        )

    def format_instructions(self) -> str:
        return self.instructions.strip()

    output_type: type[DataclassType] = BaseStackEduReduxOutput


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxCPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_c"
    preamble: str = """
Below is an extract from a C program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid C code, even if it's not educational, like boilerplate code, preprocessor directives, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., memory management). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a C course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxCSharpPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_csharp"
    preamble: str = """
Below is an extract from a C# program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid C# code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., LINQ). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a C# course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxCppPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_cpp"
    preamble: str = """
Below is an extract from a C++ program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid C++ code, even if it's not educational, like boilerplate code, preprocessor directives, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., templates). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a C++ course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxGoPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_go"
    preamble: str = """
Below is an extract from a Go program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Go code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., concurrency with goroutines). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Go course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxJavaPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_java"
    preamble: str = """
Below is an extract from a Java program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Java code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., multithreading). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Java course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxJavaScriptPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_javascript"
    preamble: str = """
Below is an extract from a JavaScript program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid JavaScript code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., asynchronous programming). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a JavaScript course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxMarkdownPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_markdown"
    preamble: str = """
Below is an extract from a Markdown document. Evaluate whether it has a high educational value and could help teach Markdown formatting. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the document contains valid Markdown syntax, even if it's not educational, like boilerplate text, plain prose, and niche formatting.

- Add another point if the document addresses practical concepts, even if it lacks explanations.

- Award a third point if the document is suitable for educational use and introduces key concepts in Markdown, even if the topic is advanced (e.g., complex table formatting). The document should be well-structured and contain some explanations.

- Give a fourth point if the document is self-contained and highly relevant to teaching Markdown. It should be similar to a tutorial or a Markdown course section.

- Grant a fifth point if the document is outstanding in its educational value and is perfectly suited for teaching Markdown. It should be well-written, easy to understand, and contain step-by-step explanations.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxPHPPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_php"
    preamble: str = """
Below is an extract from a PHP program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid PHP code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., database interactions). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a PHP course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxPythonPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_python"
    preamble: str = """
Below is an extract from a Python program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Python code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., deep learning). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Python course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxRubyPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_ruby"
    preamble: str = """
Below is an extract from a Ruby program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Ruby code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., metaprogramming). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Ruby course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxRustPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_rust"
    preamble: str = """
Below is an extract from a Rust program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Rust code, even if it's not educational, like boilerplate code, macro definitions, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., ownership and lifetimes). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Rust course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxShellPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_shell"
    preamble: str = """
Below is an extract from a Shell script. Evaluate whether it has a high educational value and could help teach scripting. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the script contains valid Shell code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the script addresses practical concepts, even if it lacks comments.

- Award a third point if the script is suitable for educational use and introduces key concepts in scripting, even if the topic is advanced (e.g., pipeline processing). The script should be well-structured and contain some comments.

- Give a fourth point if the script is self-contained and highly relevant to teaching scripting. It should be similar to a tutorial or a Shell scripting course section.

- Grant a fifth point if the script is outstanding in its educational value and is perfectly suited for teaching scripting. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxSQLPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_sql"
    preamble: str = """
Below is an extract containing SQL code. Evaluate whether it has a high educational value and could help teach SQL. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract contains valid SQL code, even if it's not educational, like boilerplate queries, schema definitions, and niche syntax.

- Add another point if the extract addresses practical concepts, even if it lacks comments.

- Award a third point if the extract is suitable for educational use and introduces key concepts in SQL, even if the topic is advanced (e.g., complex joins). The SQL should be well-structured and contain some comments.

- Give a fourth point if the extract is self-contained and highly relevant to teaching SQL. It should be similar to a tutorial or a SQL course section.

- Grant a fifth point if the extract is outstanding in its educational value and is perfectly suited for teaching SQL. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxSwiftPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_swift"
    preamble: str = """
Below is an extract from a Swift program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Swift code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., protocol-oriented programming). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Swift course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxTypeScriptPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_typescript"
    preamble: str = """
Below is an extract from a TypeScript program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid TypeScript code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., generics). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a TypeScript course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# Notebook and document formats
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxJupyterNotebookPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_jupyter_notebook"
    preamble: str = """
Below is an extract from a Jupyter Notebook. Evaluate whether it has a high educational value and could help teach coding or data science. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the notebook contains valid code cells, even if it's not educational, like boilerplate setup, package imports, and niche concepts.

- Add another point if the notebook addresses practical concepts, even if it lacks markdown explanations.

- Award a third point if the notebook is suitable for educational use and introduces key concepts in programming or data science, even if the topic is advanced (e.g., deep learning or statistical modeling). The code should be well-structured and the notebook should contain some explanatory markdown cells.

- Give a fourth point if the notebook is self-contained and highly relevant to teaching. It should be similar to a tutorial, a course assignment, or a guided walkthrough with clear narrative flow between cells.

- Grant a fifth point if the notebook is outstanding in its educational value and is perfectly suited for teaching. It should be well-written, easy to follow, and contain step-by-step explanations in markdown cells alongside well-commented code.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxReStructuredTextPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_restructuredtext"
    preamble: str = """
Below is an extract from a reStructuredText document. Evaluate whether it has a high educational value and could help teach reStructuredText formatting. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the document contains valid reStructuredText syntax, even if it's not educational, like boilerplate text, auto-generated content, and niche directives.

- Add another point if the document addresses practical concepts, even if it lacks explanations.

- Award a third point if the document is suitable for educational use and introduces key concepts in reStructuredText, even if the topic is advanced (e.g., custom directives or cross-references). The document should be well-structured and contain some explanations.

- Give a fourth point if the document is self-contained and highly relevant to teaching reStructuredText. It should be similar to a tutorial or a reStructuredText course section.

- Grant a fifth point if the document is outstanding in its educational value and is perfectly suited for teaching reStructuredText. It should be well-written, easy to understand, and contain step-by-step explanations.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxRMarkdownPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_rmarkdown"
    preamble: str = """
Below is an extract from an R Markdown document. Evaluate whether it has a high educational value and could help teach R programming or data analysis. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the document contains valid R Markdown syntax with R code chunks, even if it's not educational, like boilerplate setup, package loading, and niche concepts.

- Add another point if the document addresses practical concepts, even if it lacks narrative explanations.

- Award a third point if the document is suitable for educational use and introduces key concepts in R programming or data analysis, even if the topic is advanced (e.g., statistical modeling or parameterized reports). The code should be well-structured and the document should contain some explanatory prose.

- Give a fourth point if the document is self-contained and highly relevant to teaching. It should be similar to a tutorial, a course assignment, or a reproducible research example with clear narrative flow.

- Grant a fifth point if the document is outstanding in its educational value and is perfectly suited for teaching. It should be well-written, easy to follow, and seamlessly integrate explanatory prose with well-commented R code chunks.
"""


# =============================================================================
# Markup and styling languages
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxCSSPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_css"
    preamble: str = """
Below is an extract from a CSS stylesheet. Evaluate whether it has a high educational value and could help teach CSS. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the stylesheet contains valid CSS syntax, even if it's not educational, like boilerplate resets, vendor prefixes, and niche properties.

- Add another point if the stylesheet addresses practical concepts, even if it lacks comments.

- Award a third point if the stylesheet is suitable for educational use and introduces key concepts in CSS, even if the topic is advanced (e.g., CSS Grid). The CSS should be well-structured and contain some comments.

- Give a fourth point if the stylesheet is self-contained and highly relevant to teaching CSS. It should be similar to a tutorial or a CSS course section.

- Grant a fifth point if the stylesheet is outstanding in its educational value and is perfectly suited for teaching CSS. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxHTMLPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_html"
    preamble: str = """
Below is an extract from an HTML document. Evaluate whether it has a high educational value and could help teach HTML. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the document contains valid HTML syntax, even if it's not educational, like boilerplate markup, generated code, and niche elements.

- Add another point if the document addresses practical concepts, even if it lacks comments.

- Award a third point if the document is suitable for educational use and introduces key concepts in HTML, even if the topic is advanced (e.g., semantic elements and accessibility). The HTML should be well-structured and contain some comments.

- Give a fourth point if the document is self-contained and highly relevant to teaching HTML. It should be similar to a tutorial or an HTML course section.

- Grant a fifth point if the document is outstanding in its educational value and is perfectly suited for teaching HTML. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxSCSSPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_scss"
    preamble: str = """
Below is an extract from an SCSS stylesheet. Evaluate whether it has a high educational value and could help teach SCSS. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the stylesheet contains valid SCSS syntax, even if it's not educational, like boilerplate resets, vendor prefixes, and niche features.

- Add another point if the stylesheet addresses practical concepts, even if it lacks comments.

- Award a third point if the stylesheet is suitable for educational use and introduces key concepts in SCSS, even if the topic is advanced (e.g., mixins and functions). The SCSS should be well-structured and contain some comments.

- Give a fourth point if the stylesheet is self-contained and highly relevant to teaching SCSS. It should be similar to a tutorial or an SCSS course section.

- Grant a fifth point if the stylesheet is outstanding in its educational value and is perfectly suited for teaching SCSS. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# Lisp family
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxClojurePrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_clojure"
    preamble: str = """
Below is an extract from a Clojure program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Clojure code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., macros or transducers). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Clojure course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxCommonLispPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_common_lisp"
    preamble: str = """
Below is an extract from a Common Lisp program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Common Lisp code, even if it's not educational, like boilerplate code, system definitions, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., macros or CLOS). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Common Lisp course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxSchemePrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_scheme"
    preamble: str = """
Below is an extract from a Scheme program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Scheme code, even if it's not educational, like boilerplate code, library imports, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., continuations or macros). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Scheme course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# Functional languages
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxHaskellPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_haskell"
    preamble: str = """
Below is an extract from a Haskell program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Haskell code, even if it's not educational, like boilerplate code, language extensions, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., monads or type classes). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Haskell course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxOCamlPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_ocaml"
    preamble: str = """
Below is an extract from an OCaml program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid OCaml code, even if it's not educational, like boilerplate code, module signatures, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., functors or GADTs). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or an OCaml course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxErlangPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_erlang"
    preamble: str = """
Below is an extract from an Erlang program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Erlang code, even if it's not educational, like boilerplate code, behavior callbacks, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., OTP behaviors or message passing). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or an Erlang course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# Modern compiled languages
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxDartPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_dart"
    preamble: str = """
Below is an extract from a Dart program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Dart code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., async/await or isolates). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Dart course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxKotlinPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_kotlin"
    preamble: str = """
Below is an extract from a Kotlin program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Kotlin code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., coroutines or extension functions). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Kotlin course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxScalaPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_scala"
    preamble: str = """
Below is an extract from a Scala program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Scala code, even if it's not educational, like boilerplate code, build definitions, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., implicits or pattern matching). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Scala course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxObjectiveCPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_objective_c"
    preamble: str = """
Below is an extract from an Objective-C program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Objective-C code, even if it's not educational, like boilerplate code, framework imports, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., blocks or protocols). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or an Objective-C course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# Scientific and numerical computing
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxFortranPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_fortran"
    preamble: str = """
Below is an extract from a Fortran program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Fortran code, even if it's not educational, like boilerplate code, module interfaces, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., array operations or parallel computing). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Fortran course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxFortranFreeFormPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_fortran_free_form"
    preamble: str = """
Below is an extract from a Fortran program using free-form source format. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid modern Fortran code, even if it's not educational, like boilerplate code, module interfaces, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., array operations or coarrays). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Fortran course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxJuliaPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_julia"
    preamble: str = """
Below is an extract from a Julia program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Julia code, even if it's not educational, like boilerplate code, package imports, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., multiple dispatch or metaprogramming). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Julia course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxRPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_r"
    preamble: str = """
Below is an extract from an R program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid R code, even if it's not educational, like boilerplate code, package loading, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., data frames or vectorization). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or an R course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxMATLABPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_matlab"
    preamble: str = """
Below is an extract from a MATLAB program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid MATLAB code, even if it's not educational, like boilerplate code, toolbox calls, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., vectorization or matrix operations). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a MATLAB course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxMathematicaPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_mathematica"
    preamble: str = """
Below is an extract from a Mathematica program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Mathematica code, even if it's not educational, like boilerplate code, package imports, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., pattern matching or symbolic computation). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Mathematica course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# Scripting languages
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxLuaPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_lua"
    preamble: str = """
Below is an extract from a Lua program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Lua code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., metatables or coroutines). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Lua course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxPerlPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_perl"
    preamble: str = """
Below is an extract from a Perl program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Perl code, even if it's not educational, like boilerplate code, module imports, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., regular expressions or references). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Perl course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxTclPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_tcl"
    preamble: str = """
Below is an extract from a Tcl program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Tcl code, even if it's not educational, like boilerplate code, package requires, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., namespaces or event-driven programming). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Tcl course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxPascalPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_pascal"
    preamble: str = """
Below is an extract from a Pascal program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Pascal code, even if it's not educational, like boilerplate code, unit declarations, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., pointers or records). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Pascal course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# GPU computing
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxCUDAPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_cuda"
    preamble: str = """
Below is an extract from a CUDA program. Evaluate whether it has a high educational value and could help teach GPU programming. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid CUDA code, even if it's not educational, like boilerplate code, device queries, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in GPU programming, even if the topic is advanced (e.g., memory coalescing or kernel optimization). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching GPU programming. It should be similar to a school exercise, a tutorial, or a CUDA course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching GPU programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxOpenCLPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_opencl"
    preamble: str = """
Below is an extract from an OpenCL program. Evaluate whether it has a high educational value and could help teach GPU programming. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid OpenCL code, even if it's not educational, like boilerplate code, platform queries, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in GPU programming, even if the topic is advanced (e.g., work-groups or memory hierarchies). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching GPU programming. It should be similar to a school exercise, a tutorial, or an OpenCL course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching GPU programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# Hardware description languages
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxVerilogPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_verilog"
    preamble: str = """
Below is an extract from a Verilog module. Evaluate whether it has a high educational value and could help teach hardware design. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the module contains valid Verilog code, even if it's not educational, like boilerplate code, testbench scaffolding, and niche constructs.

- Add another point if the module addresses practical concepts, even if it lacks comments.

- Award a third point if the module is suitable for educational use and introduces key concepts in hardware design, even if the topic is advanced (e.g., finite state machines or pipelining). The code should be well-structured and contain some comments.

- Give a fourth point if the module is self-contained and highly relevant to teaching hardware design. It should be similar to a school exercise, a tutorial, or a Verilog course section.

- Grant a fifth point if the module is outstanding in its educational value and is perfectly suited for teaching hardware design. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxSystemVerilogPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_systemverilog"
    preamble: str = """
Below is an extract from a SystemVerilog module. Evaluate whether it has a high educational value and could help teach hardware design. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the module contains valid SystemVerilog code, even if it's not educational, like boilerplate code, testbench scaffolding, and niche constructs.

- Add another point if the module addresses practical concepts, even if it lacks comments.

- Award a third point if the module is suitable for educational use and introduces key concepts in hardware design, even if the topic is advanced (e.g., assertions or interfaces). The code should be well-structured and contain some comments.

- Give a fourth point if the module is self-contained and highly relevant to teaching hardware design. It should be similar to a school exercise, a tutorial, or a SystemVerilog course section.

- Grant a fifth point if the module is outstanding in its educational value and is perfectly suited for teaching hardware design. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxBluespecPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_bluespec"
    preamble: str = """
Below is an extract from a Bluespec program. Evaluate whether it has a high educational value and could help teach hardware design. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Bluespec code, even if it's not educational, like boilerplate code, interface declarations, and niche constructs.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in hardware design, even if the topic is advanced (e.g., rules, scheduling, or interfaces). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching hardware design. It should be similar to a school exercise, a tutorial, or a Bluespec course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching hardware design. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxVHDLPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_vhdl"
    preamble: str = """
Below is an extract from a VHDL module. Evaluate whether it has a high educational value and could help teach hardware design. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the module contains valid VHDL code, even if it's not educational, like boilerplate code, testbench scaffolding, and niche constructs.

- Add another point if the module addresses practical concepts, even if it lacks comments.

- Award a third point if the module is suitable for educational use and introduces key concepts in hardware design, even if the topic is advanced (e.g., processes or concurrent statements). The code should be well-structured and contain some comments.

- Give a fourth point if the module is self-contained and highly relevant to teaching hardware design. It should be similar to a school exercise, a tutorial, or a VHDL course section.

- Grant a fifth point if the module is outstanding in its educational value and is perfectly suited for teaching hardware design. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# Templating languages
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxBladePrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_blade"
    preamble: str = """
Below is an extract from a Blade template. Evaluate whether it has a high educational value and could help teach Blade templating and PHP web development. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the template contains valid Blade syntax, even if it's not educational, like boilerplate layouts, simple variable echoes, and niche directives.

- Add another point if the template addresses practical concepts, even if it lacks comments.

- Award a third point if the template is suitable for educational use and introduces key concepts in templating or PHP web development, even if the topic is advanced (e.g., components, slots, or custom directives). The template should be well-structured and contain some comments.

- Give a fourth point if the template is self-contained and highly relevant to teaching Blade or Laravel development. It should be similar to a tutorial or a Blade course section.

- Grant a fifth point if the template is outstanding in its educational value and is perfectly suited for teaching Blade templating. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxJavaServerPagesPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_java_server_pages"
    preamble: str = """
Below is an extract from a Java Server Pages (JSP) file. Evaluate whether it has a high educational value and could help teach JSP and Java web development. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the file contains valid JSP syntax, even if it's not educational, like boilerplate page directives, simple scriptlets, and niche tags.

- Add another point if the file addresses practical concepts, even if it lacks comments.

- Award a third point if the file is suitable for educational use and introduces key concepts in JSP or Java web development, even if the topic is advanced (e.g., custom tag libraries or expression language). The code should be well-structured and contain some comments.

- Give a fourth point if the file is self-contained and highly relevant to teaching JSP. It should be similar to a tutorial or a JSP course section.

- Grant a fifth point if the file is outstanding in its educational value and is perfectly suited for teaching JSP. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxVuePrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_vue"
    preamble: str = """
Below is an extract from a Vue single-file component. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the component contains valid Vue syntax, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the component addresses practical concepts, even if it lacks comments.

- Award a third point if the component is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., reactivity or the Composition API). The code should be well-structured and contain some comments.

- Give a fourth point if the component is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Vue course section.

- Grant a fifth point if the component is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""


# =============================================================================
# Data and annotation formats
# =============================================================================


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduReduxCoNLLUPrompt(BaseStackEduReduxPrompt):
    name: str = "stack_edu_redux_conllu"
    preamble: str = """
Below is an extract from a CoNLL-U file. Evaluate whether it has a high educational value and could help teach linguistic annotation or natural language processing. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the file contains valid CoNLL-U format, even if it's not educational, like auto-generated annotations, minimal feature sets, and niche language data.

- Add another point if the file addresses practical annotation concepts, even if it lacks comments or metadata explanations.

- Award a third point if the file is suitable for educational use and introduces key concepts in linguistic annotation, even if the topic is advanced (e.g., dependency relations or morphological features). The annotations should be consistent and the file should contain some explanatory comments.

- Give a fourth point if the file is self-contained and highly relevant to teaching linguistic annotation or NLP. It should be similar to a tutorial example or a course exercise with clear, well-chosen sentences.

- Grant a fifth point if the file is outstanding in its educational value and is perfectly suited for teaching linguistic annotation. It should demonstrate annotation conventions clearly, use well-chosen examples, and contain helpful comments explaining the annotation decisions.
"""
