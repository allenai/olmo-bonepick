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
class StackEduPythonPrompt(BaseStackEduPrompt):
    name: str = "stack_edu_python"
    preamble: str = """
Below is an extract from a Python program. Evaluate whether it has a high educational value and could help teach coding. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the program contains valid Python code, even if it’s not educational, like boilerplate code, configs, and niche concepts.

- Add another point if the program addresses practical concepts, even if it lacks comments.

- Award a third point if the program is suitable for educational use and introduces key concepts in programming, even if the topic is advanced (e.g., deep learning). The code should be well-structured and contain some comments.

- Give a fourth point if the program is self-contained and highly relevant to teaching programming. It should be similar to a school exercise, a tutorial, or a Python course section.

- Grant a fifth point if the program is outstanding in its educational value and is perfectly suited for teaching programming. It should be well-written, easy to understand, and contain step-by-step explanations and comments.
"""
