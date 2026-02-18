import dataclasses as dt

from bonepick.annotate.prompts import BaseAnnotationPrompt, BaseSystemPrompt, DataclassType


@dt.dataclass(frozen=True)
@BaseSystemPrompt.register
class GeneralRubricSystemPrompt(BaseSystemPrompt[str]):
    name: str = "general_rubric_system"
    instructions: str = """You are a helpful assistant that excels at perfectly categorizing text according to the rubric provided."""


@dt.dataclass(frozen=True)
class SuffixArrayDuplicationOutput(DataclassType):
    justification: str
    category: str


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class SuffixArrayDuplicationPrompt(BaseAnnotationPrompt):
    name: str = "suffix_array_duplication"
    preamble: str = """
The following snippet has been identified as duplicate text by an algorithm that uses suffix array to detect duplicate text in large web text corpora.

Label the snippet with one of the following categories:

- **boilerplate**: the snippet contains non-essential web page boilerplate, such as footers, navigation bars, terms of use/service, privacy policies, legal disclaimers, and similar elements that are not related to the main content of the page.
- **contextual**: the snippet contains text that is not part of the main prose, but provides context required to parse the rest of the page. Examples include instructions, headers, descriptors, or other similar items.
- **product_listing**: the snippet contains text that is part of a product listing, such as product names, descriptions, prices, or similar items.
- **advertisement**: the snippet contains text that is part of an advertisement, such as product tag lines, or invites to click on outgoing links.
- **prose**: the snippet contains text that is part of the main prose of the page. Examples include paragraphs, lists, how to, forum messages, or similar items.
- **other**: the snippet contains text that does not fit into any of the above categories. Use this category sparingly.

Some snippets might be truncated or incomplete; classify based on the available information. Some snippets might belong to multiple categories; pick the most representative.
    """
    instructions: str = """
After examining the extract, respond with a JSON object with the following format:

```json
{{
    "justification": "...",    # a brief justification for the category, up to 100 words
    "category": str,           # `boilerplate`, `contextual`, `product_listing`, `prose`, `advertisement`, or `other`
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

    output_type: type[DataclassType] = SuffixArrayDuplicationOutput
