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



@dt.dataclass(frozen=True)
class FinePdfOutput(DataclassType):
    justification: str
    score: int


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class FinePdfishEduRubricPrompt(BaseAnnotationPrompt):
    name: str = "finepdfish_edu"
    preamble: str = """
Below is an extract from a PDF file. Evaluate whether the extract has a high educational value and could be useful in an educational setting as teaching or reference material for students. Use the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.

- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.

- Award a third point if the extract is appropriate for educational use and covers key topics relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory overview or a basic tutorial that is suitable as reference material, but has notable limitations that make it distinct from a textbook.

- Grant a fourth point if the extract highly relevant and beneficial for educational purposes, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information. The content is coherent, focused, and valuable for structured learning.

- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for
teaching. The extract is an excellent introductory textbook on the subject matter. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.
    """
    instructions: str = """
After examining the extract, respond with a JSON object with the following format:

```json
{{
    "justification": "...",    # a brief justification for the score, up to 100 words
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

    output_type: type[DataclassType] = FinePdfOutput


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class FinePdfishDclmRubricPrompt(FinePdfishEduRubricPrompt):
    name: str = "finepdfish_dclm"
    preamble: str = """
Below is an extract from a PDF file. Evaluate whether the extract exhibits properties suitable for instruction-following or question-answering training data using the 6-point scoring system described below. Select the single score that best represents the extract's quality level:

**Score 0: Spam, Garbled, or Completely Unusable Content**
- Award 0 points for SEO spam content, promotional material with no educational value, completely garbled/corrupted text that is unreadable, random character sequences, or severely corrupted formatting that makes the content incomprehensible.

**Score 1: Simple Lists, Forms, or Minimal-Value Content**
- Award 1 point for content that has basic readable formatting but consists primarily of simple lists without context, forms, contact information, schedules, basic data tables without explanation, or other minimal-value structured content that lacks meaningful narrative or educational substance.

**Score 2: Cohesive Text Without Educational Value**
- Award 2 points if the extract contains cohesive, well-structured text that flows logically but lacks educational or instructional value. This includes meeting reports, business correspondence, letters, basic manual descriptions, administrative documents, or narrative content that doesn't teach or explain concepts.

**Score 3: Educational Content Without Q&A Structure**
- Award 3 points if the extract contains educational or informational content that could be useful for learning but doesn't follow a clear instructional format. This includes Wikipedia-style articles, research papers, academic content, encyclopedic entries, or explanatory text that presents information without explicit teaching structure.

**Score 4: Instructional Manuals and Structured Q&A**
- Award 4 points if the extract demonstrates clear instructional format with identifiable structure such as how-to guides, instruction manuals, structured question-answer pairs, problem-solution formats, or other organized pedagogical patterns. The content should be well-organized and follow recognizable educational conventions.

**Score 5: High-Quality Instructional Content with Explanations**
- Award 5 points if the extract exhibits exemplary instruction-response or question-answer properties with clear reasoning and detailed explanations. It should demonstrate thoughtful, step-by-step reasoning found in high-quality educational content like comprehensive tutorials, detailed explanations with context and reasoning, or expert-level instructional material that provides not just answers but explanatory reasoning and educational depth.
"""
    instructions: str = """
After examining the extract, respond with a JSON object with the following format:

```json
{{
    "justification": "...",    # a brief justification for the score, up to 100 words
    "score": int,              # the final score between 0 and 5 (inclusive)
}}
```
"""


@dt.dataclass(frozen=True)
@BaseAnnotationPrompt.register
class FinePdfishOcrRubricPrompt(FinePdfishEduRubricPrompt):
    name: str = "finepdfish_ocr"
    preamble: str = """
Below is an extract from a PDF file. Evaluate the quality of the document extraction using the 4-point scoring system described below. Select the single score that best represents the extraction quality level:

**Score 0: Garbage Text Present**
- Award 0 points if there are any garbage artifacts present in the text, regardless of how much legitimate content surrounds them. This includes OCR corruption like random character sequences (e.g., "7*/3./ +*/ 6- 4603"), unreadable symbol combinations, corrupted encoding artifacts, or any form of garbled text that renders portions of the document incomprehensible. Even if 90% of the text is perfectly readable, the presence of any garbage characters results in a score of 0.

**Score 1: Clear Formatting Issues**
- Award 1 point if there are no garbage characters but clear formatting problems are present. This includes broken mathematical equations or formulas that are unreadable, excessive or irregular spacing that disrupts readability, malformed tables or lists, severely corrupted line breaks, or other structural formatting issues that significantly impact the document's usability while keeping the text itself readable.

**Score 2: Minor Formatting Problems**
- Award 2 points if there are no garbage characters but minor formatting issues exist. This includes scattered extra spaces within words or sentences (e.g., "A t t h e S how"), inconsistent spacing, minor alignment issues, occasional broken line formatting, or small structural problems that don't severely impact readability but indicate imperfect extraction quality.

**Score 3: Clean Extraction**
- Award 3 points if there are no OCR garbage artifacts, no significant formatting issues, and the text extraction preserves the document's structure and readability effectively. The content should be clean, properly formatted, and easily readable with minimal to no extraction artifacts.
"""
    instructions: str = """
After examining the extract, respond with a JSON object with the following format:

```json
{{
    "justification": "...",    # a brief justification for the score, up to 100 words
    "score": int,              # the final score between 0 and 3 (inclusive)
}}
```
"""
