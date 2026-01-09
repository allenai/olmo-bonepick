import dataclasses as dt

from bonepick.annotate.prompts import BaseAnnotationPrompt, BaseSystemPrompt, DataclassType


@dt.dataclass(frozen=True)
@BaseSystemPrompt.register
class CodeSystemPrompt(BaseSystemPrompt[str]):
    name: str = "code_system"
    instructions: str = """You are a helpful coding assistant that excels in reviewing and assessing code."""


@dt.dataclass(frozen=True)
class BaseCodePrompt(BaseAnnotationPrompt[str]):
    preamble: str = """
Your task is to score the quality of a code snippet shown below, according to the rubric provided.

- The code snippet is enclosed between the markers "===== BEGIN CODE SNIPPET =====" and "===== END CODE SNIPPET ====="
- The rubric is enclosed between the markers "===== BEGIN RUBRIC =====" and "===== END RUBRIC ====="
"""

    def format_text(self, text: str, max_text_length: int | None = None) -> str:
        text = text.strip()
        if max_text_length is not None and len(text) > max_text_length:
            text = text[:max_text_length]
        return f"===== BEGIN CODE SNIPPET =====\n{text}\n===== END CODE SNIPPET =====\n"

    def format_instructions(self) -> str:
        return f"===== BEGIN RUBRIC =====\n{self.instructions.strip()}\n===== END RUBRIC =====\n"


@dt.dataclass(frozen=True)
class ClaudeCodeRubricCriterionOutput:
    explanation: str
    is_pass: bool


@dt.dataclass(frozen=True)
class ClaudeCodeRubricCriteriaOutput:
    functional_code: ClaudeCodeRubricCriterionOutput
    no_red_flags: ClaudeCodeRubricCriterionOutput
    readable: ClaudeCodeRubricCriterionOutput
    well_structured: ClaudeCodeRubricCriterionOutput
    exemplary_quality: ClaudeCodeRubricCriterionOutput


@dt.dataclass(frozen=True)
class ClaudeCodeRubricOutput:
    criteria: ClaudeCodeRubricCriteriaOutput
    overall_assessment: str
    score: int


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class ClaudeRubricCodePrompt(BaseCodePrompt):
    name: str = "claude_rubric_code"
    instructions: str = """
The following rubric is used to score a code snippet between 1 and 5. It assesses whether the code is of high quality and could be useful for teaching coding concepts, algorithms, libraries, best practices, etc. Award 1 point for each criterion that is met.

- Criterion 1 - **functional code**: Award if the code contains valid, executable logic that serves a real purpose—not just boilerplate, configuration, data, or broken fragments. Examples:
    - Yes: A function that processes input and returns a result, even if it's part of a larger system.
    - No: A config file with variable assignments but no logic.
    - No: Code with pervasive syntax errors or that is clearly incomplete mid-statement.

* Criterion 2 - **no red flags**: award unless the code has glaring issues that would immediately concern a reviewer: hardcoded secrets, obvious security holes, resource leaks, or egregiously wasteful operations. If none of these concerns apply to the code, award the point. Examples:
    - Yes: Code that opens a file using a context manager, or code that simply doesn't deal with external resources.
    - No: api_key = "sk-live-abc123..." in the source.
    - No: Opening database connections in a loop without closing them.

* Criterion 3 - **readable**: award if the code is easy to follow: clear naming, consistent style, reasonable lengths, shallow nesting, and free of clutter like dead code or excessive comments. Examples:
    - Yes: Descriptive names like user_email and calculate_tax(), with consistent indentation.
    - No: Single-letter variable names everywhere with no explanation.
    - No: Large blocks of commented-out code or debug print statements left in.

* Criterion 4 - **well-structured**: award if the code has coherent organization, uses appropriate abstractions (constants, helper functions), avoids repetition, and handles errors and edge cases rather than ignoring them. Examples:
    - Yes: Repeated logic extracted into a function; errors caught and handled meaningfully.
    - No: The same code block copy-pasted with small tweaks.
    - No: Bare except: pass hiding failures.

* Criterion 5 - **exemplary-quality**: award if the code is a strong example of good practices—well-documented where it matters, idiomatic for its language, and clear enough that a skilled developer could confidently use it as a reference. Examples:
    - Yes: Docstrings on public functions, idiomatic patterns, self-explanatory logic that needs no comments.
    - No: Clever one-liners that require deep thought to decode.
    - No: Complex logic with no explanation of intent or approach.

Respond in a json format with the following keys:
{{
    "criteria": {{
        "functional_code": {{
            "explanation": "...",   # explain why the code can be considered functional (or why not!)
            "is_pass": bool
        }},
        "no_red_flags": {{
            "explanation": "...",   # list (if any) of red flags that are present in the code
            "is_pass": bool
        }},
        "readable": {{
            "explanation": "...",   # describe what makes the code readable or not
            "is_pass": bool
        }},
        "well_structured": {{
            "explanation": "...",   # explain why the code structure is good (or why not)
            "is_pass": bool
        }},
        "exemplary_quality": {{
            "explanation": "...",   # list what makes the code exemplary (or what is missing)
            "is_pass": bool
        }}
    }},
    "overall_assessment": "...",    # a final explanation of the overall assessment of the code
    "score": int                    # the final score between 1 and 5 (inclusive); count # of "pass" values that are True
}}
"""
    output_type: type[DataclassType] = ClaudeCodeRubricOutput
