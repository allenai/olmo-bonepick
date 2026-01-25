import dataclasses as dt
import re

from bonepick.annotate.prompts import BaseAnnotationPrompt, DataclassType

from .code_rubrics import BetterTruncationCodePrompt


@dt.dataclass(frozen=True)
class StackEduOutput:
    justification: str
    score: int


@dt.dataclass(frozen=True)
class BaseStackEduPrompt(BetterTruncationCodePrompt):
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
        text = text.strip()
        if max_text_length is not None and len(text) > max_text_length:
            text = text[:max_text_length]
        return f"The extract:\n\n{text.strip()}\n\n"

    def format_instructions(self) -> str:
        return self.instructions.strip()

    output_type: type[DataclassType] = StackEduOutput


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduCPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_c"
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
class StackEduCSharpPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_csharp"
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
class StackEduCppPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_cpp"
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
class StackEduGoPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_go"
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
class StackEduJavaPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_java"
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
class StackEduJavaScriptPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_javascript"
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
class StackEduMarkdownPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_markdown"
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
class StackEduPHPPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_php"
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
class StackEduPythonPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_python"
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
class StackEduRubyPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_ruby"
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
class StackEduRustPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_rust"
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
class StackEduShellPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_shell"
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
class StackEduSQLPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_sql"
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
class StackEduSwiftPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_swift"
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
class StackEduTypeScriptPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_typescript"
    preamble: str = """
Below is an extract from a TypeScript program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid TypeScript code, even if it's not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., generics). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a TypeScript course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""
