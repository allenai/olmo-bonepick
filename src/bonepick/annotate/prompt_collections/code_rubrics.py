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

Criteria (>=3 must be true):
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

Criteria (>=3 must be true):
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

Criteria (>=3 must be true):
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

Criteria (>=3 must be true):
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

Criteria (>=3 must be true):
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

Criteria (>=3 must be true):
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

Criteria (>=3 must be true):
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

Criteria (>=3 must be true):
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

Criteria (>=3 must be true):
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

Criteria (>=3 must be true):
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



@dt.dataclass(frozen=True)
class SimplifiedCodeLevelCriterionOutput:
    explanation: str
    is_pass: bool


@dt.dataclass(frozen=True)
class SimplifiedCodeLevelsOutput:
    functional_snippet: SimplifiedCodeLevelCriterionOutput
    readable_snippet: SimplifiedCodeLevelCriterionOutput
    well_structured_snippet: SimplifiedCodeLevelCriterionOutput
    exemplary_snippet: SimplifiedCodeLevelCriterionOutput

@dt.dataclass(frozen=True)
class SimplifiedCodeOutput:
    programming_language: str
    purpose: str
    levels: SimplifiedCodeLevelsOutput
    overall_assessment: str
    score: int


@dt.dataclass(frozen=True)
class BaseCodeDocumentationPrompt(BaseAnnotationPrompt[str]):

    snippet_marker_open: str = "===== BEGIN SNIPPET ====="
    snippet_marker_close: str = "===== END SNIPPET ====="
    rubric_marker_open: str = "===== BEGIN RUBRIC ====="
    rubric_marker_close: str = "===== END RUBRIC ====="

    preamble: str = """
Your task is to score the quality of a code or documentation snippet shown below, according to the rubric provided.

- The snippet is enclosed between the markers "{snippet_marker_open}" and "{snippet_marker_close}"
- The rubric is enclosed between the markers "{rubric_marker_open}" and "{rubric_marker_close}"
"""

    def format_preamble(self) -> str:
        return self.preamble.format(
            snippet_marker_open=self.snippet_marker_open,
            snippet_marker_close=self.snippet_marker_close,
            rubric_marker_open=self.rubric_marker_open,
            rubric_marker_close=self.rubric_marker_close,
        )

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

        return f"{self.snippet_marker_open}\n{text}\n{self.snippet_marker_close}\n"

    def format_instructions(self) -> str:
        return f"{self.rubric_marker_open}\n{self.instructions.strip()}\n{self.rubric_marker_close}\n"


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class SimplifiedCodeRubricPrompt(BaseCodeDocumentationPrompt):
    name: str = "simplified_code_rubric"
    instructions: str = """
This scoring rubric is used to score the provided code or documentation snippet.

It is designed according to the following principles:
- It ranks code/documentation snippets from 0 (lowest quality; incomplete, invalid, indecipherable, etc.) to 4 (highest quality; so good it can be used as reference material).
- Scores are cumulative: you must earn all prior levels to claim the next.
- Each level has blockers (any one disqualifies) and criteria (must meet at least 3 of 5).
- Snippets might be truncated; do not penalize truncated code UNLESS a significant portion of the code is missing.

To apply the rubric, do the following:

1. Identify the programming language (python, javascript, etc.) the code is written in (this will help understand whether the code is following language-specific best practices) If it is a documentation snippet, identify the language it refers to.
2. Briefly describe the function of the code or documentation snippet. If the purpose is not clear, it might be a sign of low quality.
3. Finally, grade the snippet based on the level guidelines defined below.


# Level Guidelines

## Level 1 (Functional Snippet)

The code/documentation snippet must be minimally functional and useful; it is not fundamentally broken, incomplete, or empty.

Blockers:
- Syntax errors or corrupted text making the code non-functional.
- Mostly boilerplate, config, or data with minimal logic.
- Embedded data/blobs dominate (>25% of the file).

Criteria (>=3 must be true):
- The snippet contains valid executable code or intelligible documentation.
- Dead or commented-out code is minimal (doesn't dominate).
- No stray debug artifacts (e.g., print("here"), console.log(1), breakpoints, etc.) scattered throughout.
- Purpose is inferable: you can guess what this file does from reading it.
- File is not mostly empty, placeholder, or stub code.


## Level 2 (Readable Snippet)

The code/documentation snippet is easy to follow, well written, and free of glaring issues.

Blockers:
- Naming is systematically cryptic (mostly single letters or meaningless names) in non-trivial logic.
- Hardcoded secrets or credentials visible in the code.
- Documentation is written in poor style or grammar.
- Obvious security vulnerabilities (e.g., SQL injection via string concat, eval of user input).
- Code is procedurally generated via templates or similar mechanisms.

Criteria (>=3 must be true):
- Most identifiers (variables, functions, classes) have descriptive names.
- Consistent formatting and indentation throughout.
- Nesting is shallow: no deep pyramids of conditionals or loops.
- Lines and functions are reasonable lengths (no 500-line functions).
- No large blocks of dead, commented-out, or copy-pasted code.


## Level 3 (Well-structured Snippet)

Code snippet is well-structured, handles errors correctly, and uses appropriate abstractions; documentation snippet is well-organized and comprehensive.

Blockers:
- Abundant silent error swallowing (e.g., empty catch, except: pass).
- Copy-paste repetition of significant logic or text (e.g, same block repeated 3+ times).
- Code or documentation is procedurally generated via templates or similar mechanisms.
- Glaring inefficiencies in core paths (e.g., O(n²) when O(n) is trivial, repeated expensive operations in loops).
- Key assumptions in complex logic are completely undocumented (code) or missing (documentation).

Criteria (>=3 must be true):
- Code is decomposed into functions/classes of coherent purpose.
- When appropriate, error/exception handling logic is present; meaningful messages or recovery are provided.
- Resource handling is correct (files/connections are closed properly).
- Side effects are contained and predictable.
- Complex sections are at least minimally explained via comments (code) or prose (documentation).


## Level 4 (Exemplary Snippet)

Any code is robust, efficient, and thoughtfully designed; any documentation is well-written, clear, and comprehensive.
The snippet is suitable as a teaching reference.

Blockers:
- Resource leaks (unclosed handles, connections) in code.
- Core logic cannot be tested; documentation is not properly.
- Documentation, if present, must be comprehensive and cover all key concepts.
- Obvious inefficiencies in core paths (e.g., O(n²) when O(n) is trivial, repeated expensive operations in loops).

Criteria (>=3 must be true):
- Error/exception handling logic is robust; meaningful messages or recovery are provided.
- Docstrings or comments explain "what" and "why" for public interfaces.
- Code is idiomatic for its language—uses standard patterns well.
- Logic flows clearly enough that it could be used to teach the concepts it implements.
- Side effects are contained and predictable (I/O at edges, not scattered).


# Output Format

Return a JSON object in the following format:

```json
{{
    "programming_language": "...",   # the programming language the snippet is written in (code) or is about (documentation), all in lowercase.
    "purpose": "...",           # a brief description of the purpose of the snippet.
    "levels": {{
        "functional_snippet": {{
            "explanation": "...",   # briefly explain how the snippet meets basic checks (or why not!).
            "is_pass": bool
        }},
        "readable_snippet": {{
            "explanation": "...",   # briefly explain what makes the snippet readable or unreadable.
            "is_pass": bool
        }},
        "well_structured_snippet": {{
            "explanation": "...",   # briefly explain how the is snippet is structured, (if code) how it handles errors and uses appropriate abstractions, (if documentation) its comprehensiveness.
            "is_pass": bool
        }},
        "exemplary_snippet": {{
            "explanation": "...",   # briefly explain how the code is robust, efficient, and thoughtfully designed
            "is_pass": bool
        }},
    }},
    "overall_assessment": "...",    # a final brief explanation of the overall assessment of the snippet.
    "score": int                    # the final score between 0 and 4 (both inclusive); counts the number of "is_pass" values that are true.
}}
```

Keep all explanations brief, under 100 characters or less.
"""
    output_type: type[DataclassType] = SimplifiedCodeOutput



@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class InverseSimplifiedCodeRubricPrompt(BaseCodeDocumentationPrompt):
    name: str = "inv_simplified_code_rubric"
    rubric_marker_open: str = ""
    rubric_marker_close: str = ""
    preamble: str = """
Your task is to score the quality of a code or documentation snippet shown below, according to the rubric provided. The snippet is enclosed between the markers "{snippet_marker_open}" and "{snippet_marker_close}".

The rubric is designed according to the following principles:
- It ranks code/documentation snippets from 0 (lowest quality; incomplete, invalid, indecipherable, etc.) to 4 (highest quality; so good it can be used as reference material).
- Scores are cumulative: you must earn all prior levels to claim the next.
- Each level has blockers (any one disqualifies) and criteria (must meet at least 3 of 5).
- Snippets might be truncated; do not penalize truncated code UNLESS a significant portion of the code is missing.

To apply the rubric, do the following:

1. Identify the programming language (python, javascript, etc.) the code is written in (this will help understand whether the code is following language-specific best practices) If it is a documentation snippet, identify the language it refers to.
2. Briefly describe the function of the code or documentation snippet. If the purpose is not clear, it might be a sign of low quality.
3. Finally, grade the snippet based on the level guidelines defined below.


# Level Guidelines

## Level 1 (Functional Snippet)

The code/documentation snippet must be minimally functional and useful; it is not fundamentally broken, incomplete, or empty.

Blockers:
- Syntax errors or corrupted text making the code non-functional.
- Mostly boilerplate, config, or data with minimal logic.
- Embedded data/blobs dominate (>25% of the file).

Criteria (>=3 must be true):
- The snippet contains valid executable code or intelligible documentation.
- Dead or commented-out code is minimal (doesn't dominate).
- No stray debug artifacts (e.g., print("here"), console.log(1), breakpoints, etc.) scattered throughout.
- Purpose is inferable: you can guess what this file does from reading it.
- File is not mostly empty, placeholder, or stub code.


## Level 2 (Readable Snippet)

The code/documentation snippet is easy to follow, well written, and free of glaring issues.

Blockers:
- Naming is systematically cryptic (mostly single letters or meaningless names) in non-trivial logic.
- Hardcoded secrets or credentials visible in the code.
- Documentation is written in poor style or grammar.
- Obvious security vulnerabilities (e.g., SQL injection via string concat, eval of user input).
- Code is procedurally generated via templates or similar mechanisms.

Criteria (>=3 must be true):
- Most identifiers (variables, functions, classes) have descriptive names.
- Consistent formatting and indentation throughout.
- Nesting is shallow: no deep pyramids of conditionals or loops.
- Lines and functions are reasonable lengths (no 500-line functions).
- No large blocks of dead, commented-out, or copy-pasted code.


## Level 3 (Well-structured Snippet)

Code snippet is well-structured, handles errors correctly, and uses appropriate abstractions; documentation snippet is well-organized and comprehensive.

Blockers:
- Abundant silent error swallowing (e.g., empty catch, except: pass).
- Copy-paste repetition of significant logic or text (e.g, same block repeated 3+ times).
- Code or documentation is procedurally generated via templates or similar mechanisms.
- Glaring inefficiencies in core paths (e.g., O(n²) when O(n) is trivial, repeated expensive operations in loops).
- Key assumptions in complex logic are completely undocumented (code) or missing (documentation).

Criteria (>=3 must be true):
- Code is decomposed into functions/classes of coherent purpose.
- When appropriate, error/exception handling logic is present; meaningful messages or recovery are provided.
- Resource handling is correct (files/connections are closed properly).
- Side effects are contained and predictable.
- Complex sections are at least minimally explained via comments (code) or prose (documentation).


## Level 4 (Exemplary Snippet)

Any code is robust, efficient, and thoughtfully designed; any documentation is well-written, clear, and comprehensive.
The snippet is suitable as a teaching reference.

Blockers:
- Resource leaks (unclosed handles, connections) in code.
- Core logic cannot be tested; documentation is not properly.
- Documentation, if present, must be comprehensive and cover all key concepts.
- Obvious inefficiencies in core paths (e.g., O(n²) when O(n) is trivial, repeated expensive operations in loops).

Criteria (>=3 must be true):
- Error/exception handling logic is robust; meaningful messages or recovery are provided.
- Docstrings or comments explain "what" and "why" for public interfaces.
- Code is idiomatic for its language—uses standard patterns well.
- Logic flows clearly enough that it could be used to teach the concepts it implements.
- Side effects are contained and predictable (I/O at edges, not scattered).
"""

    instructions: str = """
# Output Format

Return a JSON object in the following format:

```json
{{
    "programming_language": "...",   # the programming language the snippet is written in (code) or is about (documentation), all in lowercase.
    "purpose": "...",           # a brief description of the purpose of the snippet.
    "levels": {{
        "functional_snippet": {{
            "explanation": "...",   # briefly explain how the snippet meets basic checks (or why not!).
            "is_pass": bool
        }},
        "readable_snippet": {{
            "explanation": "...",   # briefly explain what makes the snippet readable or unreadable.
            "is_pass": bool
        }},
        "well_structured_snippet": {{
            "explanation": "...",   # briefly explain how the is snippet is structured, (if code) how it handles errors and uses appropriate abstractions, (if documentation) its comprehensiveness.
            "is_pass": bool
        }},
        "exemplary_snippet": {{
            "explanation": "...",   # briefly explain how the code is robust, efficient, and thoughtfully designed
            "is_pass": bool
        }},
    }},
    "overall_assessment": "...",    # a final brief explanation of the overall assessment of the snippet.
    "score": int                    # the final score between 0 and 4 (both inclusive); counts the number of "is_pass" values that are true.
}}
```

Keep all explanations brief, under 100 characters or less.
"""

    def format_instructions(self) -> str:
        return f"\n\n{self.instructions.strip()}"

    output_type: type[DataclassType] = SimplifiedCodeOutput




@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class InverseSimplifiedCodeRubricV2Prompt(BaseCodeDocumentationPrompt):
    name: str = "inv_simple_codedoc_v2"
    rubric_marker_open: str = ""
    rubric_marker_close: str = ""
    preamble: str = """
# Code & Documentation Quality Scoring Rubric

This rubric scores code or documentation snippets from 0 (lowest) to 4 (highest). The snippet is enclosed between the markers "{snippet_marker_open}" and "{snippet_marker_close}".

## Principles

- **Cumulative scoring**: You must earn all prior levels to claim the next.
- **Blockers**: Any single applicable blocker disqualifies that level.
- **Criteria**: Must meet >=3 criteria to pass each level. Count from Universal + the applicable category (Code-specific OR Docs-specific).
- **Truncation**: Do not penalize truncated snippets unless a significant portion (>50%) appears missing.
- **Mixed content**: For files containing both code and documentation (e.g., docstrings, README with examples), evaluate primarily by the dominant content type, but consider both categories where relevant.

---

## Applying the Rubric

1. **Identify the language**: Determine the programming language (Python, JavaScript, etc.) or, for documentation, the language/framework it describes.
2. **Infer purpose**: Briefly describe what the snippet does. If the purpose is unclear, this may indicate low quality.
3. **Evaluate each level**: Work upward from Level 1; stop at the first level not achieved.

---

## Level 0 (Non-functional Snippet)

The snippet fails to meet any of the requirements. It is broken, empty, or unintelligible.

Examples:
- Syntax errors rendering code non-executable
- Corrupted or garbled text
- Empty or near-empty file
- Completely indecipherable purpose

---

## Level 1 (Functional Snippet)

The snippet is minimally functional and serves a discernible purpose.

### Blockers (any one disqualifies)


- Syntax errors or corrupted text making code non-executable.
- Dominated by boilerplate, config, or static data with minimal logic (>75% of content).
- Embedded binary data, base64 blobs, or similar dominate (>25% of lines).

### Criteria (>=3 must be true)

**Universal:**
- Purpose is inferable from reading the snippet.
- Not mostly empty, placeholder, or stub content.
- Content appears complete or coherently truncated (not mid-statement/sentence).

**Code-specific:**
- Contains valid, executable code.
- Dead or commented-out code is minimal (<20% of lines).
- No stray debug artifacts scattered throughout (e.g., `print("here")`, `console.log(1)`, `debugger`).
- Imports/dependencies appear reasonable (not obviously broken or circular).

**Docs-specific:**
- Documentation is intelligible and written in coherent prose.
- Covers at least one complete concept, API, or workflow.
- Free of obvious placeholder text (e.g., "TODO: write this", "Lorem ipsum").

---

## Level 2 (Readable Snippet)

The snippet is easy to follow, consistently formatted, and free of glaring issues.

### Blockers (any one disqualifies)


**Universal:**
- Hardcoded secrets, API keys, or credentials visible.

**Code-specific:**
- Naming is systematically cryptic in non-trivial logic (>50% single-letter or meaningless identifiers).
- Obvious security vulnerabilities (e.g., SQL injection via string concatenation, `eval()` on user input, path traversal).

**Docs-specific:**
- Written in poor grammar or incomprehensible style throughout.
- Factually incorrect information that would mislead readers.

### Criteria (>=3 must be true)

**Universal:**
- Consistent formatting and indentation throughout.
- No large blocks of dead, commented-out, or copy-pasted content.
- Logical flow: content is ordered sensibly (not randomly jumbled).

**Code-specific:**
- Most identifiers have descriptive, meaningful names.
- Nesting is shallow (<=4 levels of conditionals/loops in most places).
- Functions are reasonable length (<=100 lines); lines are reasonable width (<=150 chars).
- Magic numbers/strings are minimal or explained.

**Docs-specific:**
- Sections are logically organized with clear headings.
- Technical terms are used correctly and consistently.
- Grammar and spelling are largely correct (minor errors acceptable).
- Formatting aids readability (appropriate use of code blocks, lists, emphasis).

---

## Level 3 (Well-structured Snippet)

The snippet demonstrates good design: appropriate abstractions, error handling, and organization.

### Blockers (any one disqualifies)


**Universal:**
- Significant copy-paste repetition (same block of >=5 lines repeated 3+ times).
- Procedurally/mechanically generated via templates (not human-authored).

**Code-specific:**
- Abundant silent error swallowing (e.g., empty `catch`, bare `except: pass`) without justification.
- Glaring algorithmic inefficiencies in core paths (e.g., O(n²) when O(n) is trivial, repeated expensive operations in tight loops).
- Key assumptions or invariants in complex logic are completely undocumented.

**Docs-specific:**
- Critical concepts, parameters, or workflows are left unexplained.
- Instructions are ambiguous or contradictory.
- Missing prerequisites or setup steps that would leave readers stuck.

### Criteria (>=3 must be true)

**Universal:**
- Complex sections include explanatory comments (code) or clarifying prose (docs).
- Content demonstrates domain understanding (not just syntactically correct but semantically sensible).
- Avoids unnecessary complexity; solutions are proportionate to the problem.

**Code-specific:**
- Decomposed into functions/classes with coherent, single responsibilities.
- Error/exception handling is present where appropriate; provides meaningful messages or recovery.
- Resources are managed correctly (files, connections, locks are closed/released).
- Side effects are contained and predictable (I/O grouped, not scattered randomly).
- Uses appropriate data structures for the task.
- Avoids global mutable state where practical.

**Docs-specific:**
- Organized into logical sections covering distinct topics.
- Covers edge cases, limitations, or common pitfalls.
- Parameters, arguments, or configuration options are explained.
- Provides context: explains when/why to use the documented feature.
- Cross-references related concepts or links to further reading.

---

## Level 4 (Exemplary Snippet)

The snippet is robust, efficient, and thoughtfully designed. Suitable as teaching material or reference implementation.

### Blockers (any one disqualifies)


**Universal:**
- Obvious inefficiencies remain in hot paths.
- Contains outdated or deprecated approaches without acknowledgment.

**Code-specific:**
- Resource leaks (unclosed handles, connections, unreleased locks).
- Core logic is untestable (e.g., deeply coupled to global state, no separation of concerns).

**Docs-specific:**
- Incomplete coverage of key concepts, APIs, or workflows.
- No examples provided for complex operations.
- Contradicts or is inconsistent with the code it documents.

### Criteria (>=3 must be true)

**Universal:**
- Logic/content flows clearly enough to serve as a teaching example.
- Idiomatic for its language/domain — uses standard patterns, conventions, and terminology appropriately.
- Anticipates reader/user needs; answers likely follow-up questions.

**Code-specific:**
- Error handling is robust: meaningful messages, appropriate recovery, no silent failures.
- Docstrings or comments explain the "what" and "why" for public interfaces.
- Side effects are well-contained (I/O at module edges, pure functions where practical).
- Type hints or contracts are present where the language supports them.
- Defensive coding: validates inputs, handles edge cases gracefully.
- Performance-conscious: appropriate algorithms and data structures for scale.

**Docs-specific:**
- Explains both "how" and "why," not just listing facts.
- Includes concrete examples, diagrams, or code samples where helpful.
- Addresses multiple skill levels or provides progressive disclosure.
- Accurate and up-to-date with the code/system it documents.
- Comprehensive: covers the full scope of the topic without major gaps.
"""

    instructions: str = """
# Output

You should output the rubric as a JSON object.

## Output Format

```json
{{
    "programming_language": "...",   // programming language the snippet is written in (code) or is about (documentation), in lowercase.
    "purpose": "...",                // description of the purpose of the snippet.
    "levels": {{
        "functional_snippet": {{
            "explanation": "...",   // explain how the snippet meets basic checks (or why not!).
            "is_pass": bool
        }},
        "readable_snippet": {{
            "explanation": "...",   // explain what makes the snippet readable or unreadable.
            "is_pass": bool
        }},
        "well_structured_snippet": {{
            "explanation": "...",   // explain how the is snippet is structured, (if code) how it handles errors/abstractions, (if documentation) its comprehensiveness.
            "is_pass": bool
        }},
        "exemplary_snippet": {{
            "explanation": "...",   // explain how the code is robust, efficient, and thoughtfully designed
            "is_pass": bool
        }},
    }},
    "overall_assessment": "...",    // final summary of key issues of the snippet justifying the level reached.
    "score": int                    // final score between 0 and 4 (both inclusive); count # of "is_pass" values that are true.
}}
```

---

## Output Guidelines

- **Keep purpose, explanations, and overall assessment brief**: Under 100 characters or less, or 1-2 sentences at most.
- **Boolean flag per level**: `is_pass` should be true if the snippet meets the criteria for the level, false otherwise.
"""

    def format_instructions(self) -> str:
        return f"\n\n{self.instructions.strip()}"

    output_type: type[DataclassType] = SimplifiedCodeOutput



@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class InverseSimplifiedCodeRubricV3Output(InverseSimplifiedCodeRubricV2Prompt):
    name: str = "inv_simple_codedoc_v3"
    preamble: str = """
# Code & Documentation Quality Scoring Rubric

Begin with a concise checklist (3-7 bullets) describing your planned evaluation steps before proceeding. This rubric evaluates code or documentation snippets using a scale from 0 (lowest) to 4 (highest). The snippet is delimited by the markers `{snippet_marker_open}` and `{snippet_marker_close}`.

## Principles

- **Cumulative Scoring:** Each level’s requirements must be fully satisfied before advancing to the next.
- **Blockers**: Any single blocker applicable at a level disqualifies advancement to that level.
- **Criteria**: At least three criteria must be satisfied to achieve each level. Criteria are drawn from Universal plus the applicable category (Code-specific or Docs-specific).
- **Truncation**: Do not penalize truncation unless more than 50% of the snippet is missing.
- **Mixed Content**: For files combining code and documentation (e.g., docstrings, README with examples), evaluate based on the predominant content type, but consider both categories as relevant.

---

## Applying the Rubric

1. **Identify the Language**: Determine the programming language (e.g., Python, JavaScript) or, for documentation, the described language/framework.
2. **Infer Purpose**: Briefly state the snippet's intended function. If the purpose is unclear, this may indicate low quality.
3. **Evaluate Each Level**: Start at Level 1 and proceed upward, stopping at the first level the snippet does not achieve.

After scoring, validate the assigned level in 1-2 lines, explicitly stating whether all required blockers and criteria were considered and if the result matches the rubric standards. Self-correct if misalignment is detected.

---

# Level Guidelines

## Level 0 (Non-functional Snippet)

The snippet fails to meet any requirements and is considered broken, empty, or unintelligible.

**Examples:**
- Syntax errors rendering code non-executable
- Corrupted or garbled text
- Empty or near-empty file
- Indecipherable purpose

---

## Level 1 (Functional Snippet)

The snippet is minimally functional and has a discernible purpose.

### Blockers (any one disqualifies)

**Universal:**
- Syntax errors or corrupted text making code non-executable
- Predominant boilerplate, configuration, or static data with minimal logic (>75% of content)
- Dominance of embedded binary data, base64 blobs, or similar (>25% of lines)

### Criteria (>=3 must be true)

**Universal:**
- Purpose inferable from the snippet
- Not mostly empty, placeholder, or stub content
- Appears complete or is coherently truncated (not mid-statement/sentence)

**Code-specific:**
- Contains valid, executable code
- Minimal dead or commented-out code (<20% of lines)
- No stray debug artifacts scattered throughout (e.g., `print("here")`, `console.log(1)`, `debugger`)
- Reasonable imports/dependencies (not obviously broken or circular)

**Docs-specific:**
- Intelligible documentation in coherent prose
- Covers at least one complete concept, API, or workflow
- Free of placeholder text (e.g., "TODO: write this", "Lorem ipsum")

---

## Level 2 (Readable Snippet)

The snippet is easy to follow, consistently formatted, and free of major issues.

### Blockers (any one disqualifies)

**Universal:**
- Exposed hardcoded secrets, API keys, or credentials

**Code-specific:**
- Systematically cryptic naming in non-trivial logic (>50% single-letter or meaningless identifiers)
- Obvious security vulnerabilities (e.g., SQL injection, `eval()` on user input, path traversal)

**Docs-specific:**
- Poor grammar or incomprehensible style throughout
- Factually incorrect information liable to mislead

### Criteria (>=3 must be true)

**Universal:**
- Consistent formatting and indentation
- No large blocks of dead, commented-out, or copy-pasted content
- Logical and sensible ordering of content

**Code-specific:**
- Descriptive and meaningful identifiers in most places
- Shallow nesting (4-5 levels of conditionals/loops)
- Functions are of reasonable length (<=100 lines); lines not overly wide (<=150 chars)
- Magic numbers/strings minimized or explained

**Docs-specific:**
- Logically organized sections with clear headings
- Correct and consistent use of technical terms
- Largely correct grammar and spelling (minor errors tolerated)
- Formatting aids readability (code blocks, lists, emphasis)

---

## Level 3 (Well-structured Snippet)

The snippet shows good design, appropriate abstractions, error handling, and organization.

### Blockers (any one disqualifies)

**Universal:**
- Significant copy-paste repetition (same block of >=5 lines repeated 3+ times)
- Procedural or mechanically generated content not authored by a human

**Code-specific:**
- Pervasive silent error swallowing (e.g., empty `catch`, bare `except: pass`) without justification
- Glaring algorithmic inefficiencies in the core logic
- Undocumented key assumptions/invariants in complex logic

**Docs-specific:**
- Crucial concepts, parameters, or workflows unexplained
- Ambiguous or contradictory instructions
- Missing prerequisites or setup steps needed to proceed

### Criteria (>=3 must be true)

**Universal:**
- Complex sections contain explanatory comments (code) or clarifying prose (docs)
- Demonstrates semantic understanding of the domain
- Avoids unnecessary complexity; solution fits the problem

**Code-specific:**
- Logical decomposition into functions/classes with clear responsibilities
- Appropriate error/exception handling, with meaningful recovery or messaging
- Correct management of resources (files, connections, locks)
- Predictable and contained side effects (I/O grouped, not scattered)
- Uses suitable data structures
- Avoids global mutable state when feasible

**Docs-specific:**
- Logically organized sections for distinct topics
- Coverage of edge cases, limitations, or common pitfalls
- Explanation of parameters, arguments, or configs
- Provides context on usage and intent of features
- Cross-references or links to related concepts/resources

---

## Level 4 (Exemplary Snippet)

The snippet is robust, efficient, and thoughtfully designed, suitable as a teaching example or reference.

### Blockers (any one disqualifies)

**Universal:**
- Obvious inefficiencies in hot paths
- Use of outdated or deprecated methods without acknowledgment

**Code-specific:**
- Resource leaks (unclosed handles, connections, unreleased locks)
- Untestable core logic (e.g., excessive coupling, no separation of concerns)

**Docs-specific:**
- Incomplete coverage of key concepts, APIs, or workflows
- No example for complex operations
- Inconsistency or contradiction with the code documented

### Criteria (>=3 must be true)

**Universal:**
- Clear, well-structured flow suitable for teaching
- Idiomatic use of language/domain conventions and patterns
- Anticipates user/reader needs, answering likely follow-up questions

**Code-specific:**
- Robust error handling: clear messages, effective recovery, no silent failures
- Public interfaces documented with relevant docstrings/comments on "what" and "why"
- Well-contained side effects (I/O at module boundaries, purity maintained where practical)
- Type hints or contracts used where supported
- Defensive: validates all inputs, gracefully handles edge cases
- Performance-conscious: appropriate algorithms and data structures for expected scale

**Docs-specific:**
- Covers both the "how" and "why" of the topic
- Provides examples, diagrams, or code samples as needed
- Addresses a range of skill levels or provides progressive disclosure
- Content is accurate and kept up-to-date with the code/system
- Comprehensive treatment without significant omissions
"""

    instructions: str = """
# Output

You should output the rubric as a JSON object.

## Output Format

```json
{{
    "programming_language": "...",   // programming language the snippet is written in (code) or is about (documentation), in lowercase.
    "purpose": "...",                // description of the purpose of the snippet.
    "levels": {{
        "functional_snippet": {{
            "explanation": "...",   // explain how the snippet meets basic checks (or why not!).
            "is_pass": bool
        }},
        "readable_snippet": {{
            "explanation": "...",   // explain what makes the snippet readable or unreadable.
            "is_pass": bool
        }},
        "well_structured_snippet": {{
            "explanation": "...",   // explain how the is snippet is structured, (if code) how it handles errors/abstractions, (if documentation) its comprehensiveness.
            "is_pass": bool
        }},
        "exemplary_snippet": {{
            "explanation": "...",   // explain how the code is robust, efficient, and thoughtfully designed
            "is_pass": bool
        }},
    }},
    "overall_assessment": "...",    // final summary of key issues of the snippet justifying the level reached.
    "score": int                    // final score between 0 and 4 (both inclusive); count # of "is_pass" values that are true.
}}
```

---

## Output Guidelines

- **Keep purpose, explanations, and overall assessment brief**: Under 100 characters or less, or 1-2 sentences at most.
- **Boolean flag per level**: `is_pass` should be true if the snippet meets the criteria for the level, false otherwise.
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class InverseSimplifiedCodeRubricV3EasyOutput(InverseSimplifiedCodeRubricV3Output):
    name: str = "inv_simple_codedoc_v3_easy"
    preamble: str = """
# Code & Documentation Quality Scoring Rubric (0–4)

The snippet is between {snippet_marker_open} and {snippet_marker_close}.

## Principles
- **Cumulative**: Must pass Level 1 to claim Level 2, etc.
- **Blockers**: Any applicable blocker fails that level.
- **Pass rule**: For each level, meet >=3 criteria, counting Universal + applicable category (Code OR Docs).
- **Truncation**: Don’t penalize truncation unless >50% seems missing.
- **Mixed content**: Score by dominant type (code vs docs), but consider both when relevant.

---

## Applying the Rubric

1. Identify language / framework.
2. Infer purpose (if unclear, likely low quality).
3. Start at Level 1 and stop at first level not achieved.

⸻

# Level Guidelines

## Level 0 (Non-functional Snippet)

Broken, empty, garbled, or purpose is not meaningfully interpretable.

⸻

## Level 1 (Functional Snippet)

Minimally functional with a discernible purpose.

### Blockers (any one disqualifies)

**Universal:**
- Syntax/corruption prevents execution or comprehension.
- Mostly boilerplate/config/static data with minimal logic (>75%).
- Binary/base64-like blobs dominate (>25% lines).

### Criteria (>=3 must be true)

**Universal:**
- Purpose is inferable.
- Not mostly empty/placeholder/stub.
- Complete or coherently truncated (not mid-statement/sentence).

**Code-specific:**
- Contains executable logic (not just declarations).
- Dead/commented-out code is minor (<20%).
- Few/no scattered debug artifacts (print, console.log, debugger).
- Imports/deps look reasonable (not obviously broken/circular).

**Docs-specific:**
- Coherent prose; intelligible.
- Covers at least one complete concept/API/workflow.
- No obvious placeholders (“TODO”, lorem ipsum).

---

## Level 2 (Readable Snippet)

Easy to follow; consistently formatted; no glaring issues.

### Blockers (any one disqualifies)

**Universal:**
- Hardcoded secrets/credentials (keys, tokens, passwords).

**Code-specific:**
- Systematically cryptic naming in non-trivial logic (>50% meaningless/single-letter identifiers).
- Obvious security hazards (e.g., eval on user input, SQL injection patterns, path traversal).

**Docs-specific:**
- Incomprehensible grammar/style throughout.
- Material factual errors likely to mislead.

### Criteria (>=3 must be true)

**Universal:**
- Consistent formatting/indentation.
- No large dead/copy-pasted blocks.
- Sensible ordering / clear flow.

**Code-specific:**
- Mostly descriptive names.
- Nesting generally shallow (≈ <=4 levels).
- Functions/lines reasonably sized (≈ <=100 lines per function; <=150 chars/line).
- Few “magic” constants, or they’re explained.

**Docs-specific:**
- Clear headings / logical sections.
- Terms used consistently and correctly.
- Mostly correct grammar/spelling.
- Formatting helps (lists, code blocks, emphasis).

---

## Level 3 (Well-structured Snippet)

Good design choices: abstraction, error handling, organization.

### Blockers (any one disqualifies)

**Universal:**
- Significant copy-paste repetition (>=5-line block repeated 3+ times).
- Clearly template/mechanically generated (not human-authored).

**Code-specific:**
- Silent error swallowing (empty catch, bare except: pass) without justification.
- Glaring avoidable inefficiency in core paths (e.g., trivial O(n) vs O(n²)).
- Complex logic has key assumptions/invariants entirely undocumented.

**Docs-specific:**
- Critical concepts/params/workflows missing.
- Instructions ambiguous/contradictory.
- Missing prerequisites/setup that would block a user.

### Criteria (>=3 must be true)

**Universal:**
- Complex parts include clarifying comments/prose.
- Demonstrates domain understanding (semantically sensible).
- Complexity is proportionate; avoids unnecessary cleverness.

**Code-specific:**
- Decomposed into coherent functions/classes (single responsibilities).
- Error/exception handling where appropriate; meaningful messages/recovery.
- Correct resource management (close/release files/conns/locks).
- Side effects are contained/predictable (I/O not scattered).
- Appropriate data structures; avoids global mutable state where practical.

**Docs-specific:**
- Sections map to distinct topics.
- Mentions edge cases/limits/pitfalls.
- Explains parameters/arguments/config.
- Provides “when/why” context.
- Cross-references related concepts/resources where helpful.

---

## Level 4 (Exemplary Snippet)

Robust, efficient, and suitable as teaching/reference material.

### Blockers (any one disqualifies)

**Universal:**
- Obvious hot-path inefficiencies remain.
- Uses deprecated/outdated approaches without noting it.

**Code-specific:**
- Resource leaks in practice.
- Core logic is effectively untestable due to tight coupling/global state.

**Docs-specific:**
- Key coverage gaps for the intended scope.
- No examples for non-trivial operations.
- Contradicts the code/system it documents.

### Criteria (>=3 must be true)

**Universal:**
- Clear enough to teach from (conceptual flow is strong).
- Idiomatic for the language/domain.
- Anticipates common reader/user questions.

**Code-specific:**
- Robust error handling (no silent failures; appropriate recovery).
- Public interfaces have docstrings/comments explaining what/why.
- Side effects pushed to edges; pure-ish core where practical.
- Type hints/contracts when supported.
- Defensive handling of inputs/edge cases.
- Performance-conscious algorithms/data structures.

**Docs-specific:**
- Explains both “how” and “why.”
- Concrete examples (snippets/commands/diagrams as appropriate).
- Progressive disclosure or guidance for different skill levels.
- Accurate and current vs the system/code.
- Comprehensive for the stated scope (no major gaps).
"""

    instructions: str = """
# Output

Return JSON only (no markdown, no extra text).

## Required fields

- `programming_language`: lowercase string (e.g., "python", "javascript", "markdown").
- `purpose`: 1 short sentence.
- `levels`: object with the following keys: `functional_snippet`, `readable_snippet`, `well_structured_snippet`, `exemplary_snippet`.
  - Each level object must have the following keys:
    - `is_pass`: boolean
    - `explanation`: 1 short sentence
- `overall_assessment`: 1–2 sentences max.
- score: integer between 0 and 4 (both inclusive), equal to the number of `is_pass: true` across the 4 levels.

## JSON template

```json
{
  "programming_language": "...",
  "purpose": "...",
  "levels": {
    "functional_snippet": { "explanation": "...", "is_pass": ... },
    "readable_snippet": { "explanation": "...", "is_pass": ... },
    "well_structured_snippet": { "explanation": "...", "is_pass": ... },
    "exemplary_snippet": { "explanation": "...", "is_pass": ... }
  },
  "overall_assessment": "...",
  "score": ...
}
```

## Style rules
- Keep purpose / each explanation concise (prefer ≤1 sentence).
- If a higher level fails, lower levels may still pass (cumulative applies to the final score, not the booleans themselves).
- `score` must match the booleans exactly (count trues).
"""
