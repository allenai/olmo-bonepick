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
The following rubric is used to score a code snippet between 1 and 5. It assesses whether the code is of high quality and could be useful for teaching coding concepts, algorithms, libraries, best practices, etc.

Award 1 point for each criterion that is met.

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


@dt.dataclass(frozen=True)
class ClaudeProgressiveCodeRubricLevelOutput:
    explanation: str
    is_pass: bool


@dt.dataclass(frozen=True)
class ClaudeProgressiveCodeRubricLevelsOutput:
    functional_code: ClaudeProgressiveCodeRubricLevelOutput
    readable_code: ClaudeProgressiveCodeRubricLevelOutput
    structured_code: ClaudeProgressiveCodeRubricLevelOutput
    well_engineered_code: ClaudeProgressiveCodeRubricLevelOutput
    excellent_code: ClaudeProgressiveCodeRubricLevelOutput


@dt.dataclass(frozen=True)
class ClaudeProgressiveCodeRubricOutput:
    programming_language: str
    code_purpose: str
    levels: ClaudeProgressiveCodeRubricLevelsOutput
    overall_assessment: str
    score: int


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class ClaudeProgressiveRubricCodePrompt(BaseCodePrompt):
    name: str = "claude_progressive_rubric_code"
    instructions: str = """
The following rubric is used to score the code snippet above between 1 and 5 (inclusive). It assesses whether the code is of high quality and could be useful for teaching coding concepts, algorithms, libraries, best practices, etc. Scoring guide:

- Scores are cumulative: you must earn all prior levels to claim the next.
- Each level has blockers (any one disqualifies) and criteria (must meet at least 3 of 5).
- If reviewing an incomplete extract, cap the score at 3 unless the excerpt shows the file's structure clearly.

First, determine what programming language (python, javascript, etc.) the code is written in (this will help understand whether the code is following language-specific best practices), and briefly describe its function. Then, determine the level of the code based on the following criteria.

## Level 1 (Functional Code)

The code must be functional and not fundamentally broken or empty.

Blockers:
- Syntax errors or corrupted text making the code non-functional.
- Mostly boilerplate, config, or data with minimal logic.
- Embedded data/blobs dominate (>25% of the file).

Criteria (≥3 must be true):
- Contains valid, executable code that could plausibly run.
- Dead or commented-out code is minimal (doesn't dominate).
- No stray debug artifacts (e.g., print("here"), console.log(1)) scattered throughout.
- Purpose is inferable: you can guess what this file does from reading it.
- File is not mostly empty, placeholder, or stub code.


## Level 2 (Readable Code)

The code is easy to follow and free of glaring issues.

Blockers:
- Naming is systematically cryptic (mostly single letters or meaningless names) in non-trivial logic.
- Hardcoded secrets or credentials visible in the code.
- Obvious security vulnerabilities (e.g., SQL injection via string concat, eval of user input).

Criteria (≥3 must be true):
- Most identifiers (variables, functions, classes) have descriptive names.
- Consistent formatting and indentation throughout.
- Nesting is shallow: no deep pyramids of conditionals or loops.
- Lines and functions are reasonable lengths (no 500-line functions).
- No large blocks of dead, commented-out, or copy-pasted code.


## Level 3 (Structured Code)

The code handles errors and uses appropriate abstractions.

Blockers:
- Silent error swallowing in multiple places (e.g., empty catch, except: pass).
- Copy-paste repetition of significant logic (same block repeated 3+ times).

Criteria (≥3 must be true):
- Code is decomposed into functions/classes of coherent purpose.
- Errors are handled with meaningful messages or recovery, not ignored.
- Magic numbers/strings are replaced with named constants where it matters.
- Resource handling is correct (files/connections closed properly).
- Public interface is distinguishable from internal helpers.


## Level 4 (Well-Engineered Code)

The code is robust, efficient, and thoughtfully designed.

Blockers:
- Key assumptions or invariants in complex logic are completely undocumented.
- Obvious inefficiencies in core paths (e.g., O(n²) when O(n) is trivial, repeated expensive operations in loops).

Criteria (≥3 must be true):
- Core logic is testable: can be exercised without elaborate setup or mocking.
- Side effects are contained and predictable (I/O at edges, not scattered).
- Complex sections have brief explanatory comments.
- Consistent conventions throughout (naming, error handling, structure).
- Preconditions or edge cases are checked or documented.


## Level 5 (Excellent Code)

The code is exemplary—suitable as a teaching reference.

Blockers:
- Any resource leaks in core paths (unclosed handles, connections).
- Observable bugs in core logic.

Criteria (≥3 must be true):
- Docstrings or comments explain "what" and "why" for public interfaces.
- Edge cases, failure modes, or limitations are documented.
- Code is idiomatic for its language—uses standard patterns well.
- A skilled developer could confidently use this as a reference.
- Logic flows clearly enough that it could be used to teach the concepts it implements.


# Output format

Respond in a json format with the following keys:
{{
    "programming_language": "...",   # the programming language the code is written in (all lowercase)
    "code_purpose": "...",           # a brief description of the purpose of the code (roughly <= 100 characters)
    "levels": {{
        "functional_code": {{
            "explanation": "...",   # briefly explain how the code meets basic checks (or why not!)
            "is_pass": bool
        }},
        "readable_code": {{
            "explanation": "...",   # briefly explain what makes the code readable or unreadable
            "is_pass": bool
        }},
        "structured_code": {{
            "explanation": "...",   # briefly explain how code manages errors and handles abstractions
            "is_pass": bool
        }},
        "well_engineered_code": {{
            "explanation": "...",   # briefly explain how the code is robust, efficient, and thoughtfully designed
            "is_pass": bool
        }},
        "excellent_code": {{
            "explanation": "...",   # briefly explain how the code is exemplary—suitable as a teaching reference
            "is_pass": bool
        }}
    }},
    "overall_assessment": "...",    # a final brief explanation of the overall assessment of the code
    "score": int                    # the final score between 1 and 5 (inclusive); count # of "pass" values that are true
}}

"""
    output_type: type[DataclassType] = ClaudeProgressiveCodeRubricOutput



@dt.dataclass(frozen=True)
class BetterTruncationCodePrompt(BaseCodePrompt):
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

        return f"===== BEGIN CODE SNIPPET =====\n{text}\n===== END CODE SNIPPET =====\n"



@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class ClaudeProgressiveRubricCodeV2Prompt(BetterTruncationCodePrompt):

    name: str = "claude_progressive_rubric_code_v2"
    instructions: str = """
The following rubric is used to score the code snippet above between 1 and 5 (inclusive). It assesses whether the code is of high quality and could be useful for teaching coding concepts, algorithms, libraries, best practices, etc. Scoring guide:

- Scores are cumulative: you must earn all prior levels to claim the next.
- Each level has blockers (any one disqualifies) and criteria (must meet at least 3 of 5).
- Do not penalize truncated code UNLESS a significant portion of the code is missing; in that case, the score should be 3 or lower.

First, determine what programming language (python, javascript, etc.) the code is written in (this will help understand whether the code is following language-specific best practices), and briefly describe its function. Then, determine the level of the code based on the following criteria.

## Level 1 (Functional Code)

The code must be functional and not fundamentally broken or empty.

Blockers:
- Syntax errors or corrupted text making the code non-functional.
- Mostly boilerplate, config, or data with minimal logic.
- Embedded data/blobs dominate (>25% of the file).

Criteria (≥3 must be true):
- Contains valid, executable code that could plausibly run.
- Dead or commented-out code is minimal (doesn't dominate).
- No stray debug artifacts (e.g., print("here"), console.log(1)) scattered throughout.
- Purpose is inferable: you can guess what this file does from reading it.
- File is not mostly empty, placeholder, or stub code.


## Level 2 (Readable Code)

The code is easy to follow and free of glaring issues.

Blockers:
- Naming is systematically cryptic (mostly single letters or meaningless names) in non-trivial logic.
- Hardcoded secrets or credentials visible in the code.
- Obvious security vulnerabilities (e.g., SQL injection via string concat, eval of user input).

Criteria (≥3 must be true):
- Most identifiers (variables, functions, classes) have descriptive names.
- Consistent formatting and indentation throughout.
- Nesting is shallow: no deep pyramids of conditionals or loops.
- Lines and functions are reasonable lengths (no 500-line functions).
- No large blocks of dead, commented-out, or copy-pasted code.


## Level 3 (Structured Code)

The code handles errors and uses appropriate abstractions.

Blockers:
- Silent error swallowing in multiple places (e.g., empty catch, except: pass).
- Copy-paste repetition of significant logic (same block repeated 3+ times).
- Code is procedurally generated via templates or similar mechanisms.

Criteria (≥3 must be true):
- Code is decomposed into functions/classes of coherent purpose.
- Minimal error/exception handling when necessary; meaningful messages or recovery are provided.
- Magic numbers/strings are replaced with named constants where it matters.
- Resource handling is correct (files/connections are closed properly).
- Public interface is distinguishable from internal helpers.


## Level 4 (Well-Engineered Code)

The code is robust, efficient, and thoughtfully designed.

Blockers:
- Key assumptions or invariants in complex logic are completely undocumented.
- Core logic cannot be tested.
- Obvious inefficiencies in core paths (e.g., O(n²) when O(n) is trivial, repeated expensive operations in loops).

Criteria (≥3 must be true):
- Most errors/exceptions are handled with meaningful messages or recovery, not ignored.
- Side effects are contained and predictable (I/O at edges, not scattered).
- Complex sections have brief explanatory comments.
- Consistent conventions throughout (naming, error handling, structure).
- Preconditions or edge cases are checked or documented.


## Level 5 (Excellent Code)

The code is exemplary—suitable as a teaching reference.

Blockers:
- Any resource leaks in core paths (unclosed handles, connections).
- Observable bugs in core logic.

Criteria (≥3 must be true):
- Docstrings or comments explain "what" and "why" for public interfaces.
- Edge cases, failure modes, or limitations are documented.
- Code is idiomatic for its language—uses standard patterns well.
- Developers could use this as a reference.
- Logic flows clearly enough that it could be used to teach the concepts it implements.


# Output format

Respond in a json format with the following keys:

```json
{{
    "programming_language": "...",   # the programming language the code is written in (all lowercase)
    "code_purpose": "...",           # a brief description of the purpose of the code.
    "levels": {{
        "functional_code": {{
            "explanation": "...",   # briefly explain how the code meets basic checks (or why not!)
            "is_pass": bool
        }},
        "readable_code": {{
            "explanation": "...",   # briefly explain what makes the code readable or unreadable
            "is_pass": bool
        }},
        "structured_code": {{
            "explanation": "...",   # briefly explain how code manages errors and handles abstractions
            "is_pass": bool
        }},
        "well_engineered_code": {{
            "explanation": "...",   # briefly explain how the code is robust, efficient, and thoughtfully designed
            "is_pass": bool
        }},
        "excellent_code": {{
            "explanation": "...",   # briefly explain how the code is exemplary—suitable as a teaching reference
            "is_pass": bool
        }}
    }},
    "overall_assessment": "...",    # a final brief explanation of the overall assessment of the code
    "score": int                    # the final score between 1 and 5 (inclusive); count # of "pass" values that are true
}}
```

Keep all explanations brief, under 100 characters or less.
"""
    output_type: type[DataclassType] = ClaudeProgressiveCodeRubricOutput


@dt.dataclass(frozen=True)
class StackEduOutput:
    justification: str
    score: int


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class StackEduPythonPrompt(BetterTruncationCodePrompt):
    name: str = "stack_edu_python"
    preamble: str = """
Below is an extract from a Python program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Python code, even if it’s not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., deep learning). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Python course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""

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
