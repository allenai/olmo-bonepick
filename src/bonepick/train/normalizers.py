from functools import partial
import re
from typing import Callable, Generic, TypeVar
import unicodedata
from math import sqrt
from plsfix import fix_text
from anyascii import anyascii

from tokenizers import Tokenizer


__all__ = [
    "register_normalizer",
    "get_normalizer",
    "list_normalizers",
]


T = TypeVar("T", bound="BaseRowNormalizer")


NORMALIZER_REGISTRY: dict[str, type["BaseRowNormalizer"]] = {}


def cut_and_ftfy_text(text: str, max_length: int = 2**20) -> str:
    if len(text) > max_length:
        next_newline = text.find("\n", max_length)
        text = text[:max_length] if next_newline == -1 else text[:next_newline]
    try:
        # split first, then fix, then join
        fragments = re.split(r"\b", text)
        fixed_fragments = [fix_text(t) for t in fragments]
        return "".join(fixed_fragments)
    except Exception:
        if max_length < 0:
            return text
        return cut_and_ftfy_text(text, max_length=max_length // 2)


def cut_and_ascii_text(text: str, max_length: int = 2**20) -> str:
    if len(text) > max_length:
        next_newline = text.find("\n", max_length)
        text = text[:max_length] if next_newline == -1 else text[:next_newline]
    try:
        text_fragments = re.split(r"\b", text)
        ascii_fragments = [anyascii(t) for t in text_fragments]
        return "".join(ascii_fragments)
    except Exception:
        if max_length < 0:
            return text
        return cut_and_ascii_text(text, max_length=max_length // 2)


def register_normalizer(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(normalizer_cls: type[T]) -> type[T]:
        if name in NORMALIZER_REGISTRY and normalizer_cls is not NORMALIZER_REGISTRY[name]:
            raise ValueError(f"Normalizer {name} already registered")
        NORMALIZER_REGISTRY[name] = normalizer_cls
        return normalizer_cls

    return decorator


def get_normalizer(name: str) -> "BaseRowNormalizer":
    if name not in NORMALIZER_REGISTRY:
        raise ValueError(f"Normalizer {name} not found")
    return NORMALIZER_REGISTRY[name]()


def list_normalizers() -> list[str]:
    return list(NORMALIZER_REGISTRY.keys())


class BaseRowNormalizer(Generic[T]):
    def __init__(self):
        pass

    def normalize(self, text: str) -> str:
        raise NotImplementedError


@register_normalizer("whitespace")
class WhitespaceNormalizer(BaseRowNormalizer):
    def __init__(self):
        self.whitespace_re = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        return self.whitespace_re.sub(" ", text).strip()


@register_normalizer("plsfix")
class PLSFixNormalizer(BaseRowNormalizer):
    def normalize(self, text: str) -> str:
        return cut_and_ftfy_text(text)


@register_normalizer("tokenizer")
class TokenizerNormalizer(BaseRowNormalizer):
    def __init__(self):
        self.tokenizer = Tokenizer.from_pretrained("allenai/dolma2-tokenizer")

    def normalize(self, text: str) -> str:
        cleaned_text = cut_and_ftfy_text(text)
        tokens = self.tokenizer.encode(cleaned_text)
        return " ".join(tokens.tokens)


@register_normalizer("ultrafine")
class UltraFineWebNormalizer(BaseRowNormalizer):
    def __init__(self):
        self.tokenizer = Tokenizer.from_pretrained("allenai/Ultra-FineWeb-tokenizer")

    def normalize(self, text: str) -> str:
        # 1. remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 2. lower the content
        text = text.lower()

        # 3. remove diacritics
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")

        # 4. word segmentation
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        single_text_list = []
        for token_id in token_ids.ids:
            curr_text = self.tokenizer.decode([token_id])
            single_text_list.append(curr_text)

        text = " ".join(single_text_list)

        # 5. keep escape chars, \n, \t, \r -> \\n, \\t, \\r,
        # which will saved as \n, \t, \r in txt file.
        text = re.sub(r"\n", "\\\\n", text)
        text = re.sub(r"\r", "\\\\r", text)
        text = re.sub(r"\t", "\\\\t", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()

        return text


@register_normalizer("ultrafine-plus")
class UltraFineWebPlusNormalizer(BaseRowNormalizer):
    def __init__(self):
        self.remove_extra_newlines_re = re.compile(r"\n{3,}")
        self.space_before_special_re = re.compile(r"(\n\r\t)")
        self.whitespace_re = re.compile(r"[^\S\n\r\t]+")
        self.tokenizer = Tokenizer.from_pretrained("allenai/dolma2-tokenizer")  # equivalent to cl100k_base

    def normalize(self, text: str) -> str:
        # 1. fix text with faster verson of ftfy
        text = cut_and_ftfy_text(text)

        # 2. remove multiple newlines
        text = self.remove_extra_newlines_re.sub("\n\n", text)

        # 3. lower the text
        text = text.lower()

        # 4. remove diacritics
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")

        # 5. add spacing around special characters
        text = self.space_before_special_re.sub(" \\1 ", text)

        # 6. first removal of extra whitespace
        text = self.whitespace_re.sub(" ", text)

        # ?. Convert to ASCII (including romanization of non-ASCII characters)
        # skipping for now bc it's unproven
        # text = cut_and_ascii_text(text)

        # 7. word segmentation
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        text = " ".join(tokens.tokens)

        # 8. second removal of extra whitespace
        text = self.whitespace_re.sub(" ", text)

        return text.strip()


@register_normalizer("potion")
class PotionNormalizer(BaseRowNormalizer):
    def __init__(self):
        self.remove_extra_newlines_re = re.compile(r"\n{3,}")
        self.space_before_special_re = re.compile(r"(\n\r\t)")
        self.whitespace_re = re.compile(r"[^\S\n\r\t]+")

    def normalize(self, text: str) -> str:
        # 1. fix text
        text = cut_and_ftfy_text(text)

        # 2. remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 3. lower the content
        text = text.lower()

        # 4. remove diacritics
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")

        # 5. escape spacing characters
        text = re.sub(r"\r?\n\r?\n", "¶", text)  # end of paragraph marker for multiple newlines
        text = re.sub(r"\r?\n", "·", text)  # just cdot for single newlines
        text = re.sub(r"\t", "↦", text)  # use ↦ for tabs
        text = re.sub(r"\s+", " ", text)  # remove extra whitespace
        text = text.strip()

        return text


@register_normalizer("potion-code")
class PotionCodeNormalizer(BaseRowNormalizer):
    def __init__(self):
        self.remove_extra_newlines_re = re.compile(r"\n{3,}")
        self.space_before_special_re = re.compile(r"(\n\r\t)")
        self.whitespace_re = re.compile(r"[^\S\n\r\t]+")

        # now replace leading spaces with most likely tab indentation

    def _replace_leading_spaces(self, match: re.Match, indentation: int) -> str:
        spaces = match.group(1)
        num_indents = len(spaces) // indentation
        remainder = len(spaces) % indentation
        return "\t" * num_indents + " " * remainder

    def _replace_tabs_with_spaces(self, text: str) -> str:
        """Replace tabs with spaces in the text."""

        indentation_lengths = [
            len(match.group(1))
            for line in text.split("\n")
            # at least 2 spaces
            if (match := re.match(r"^( {2,})", line)) is not None
        ]

        if len(indentation_lengths) == 0:
            # likely uses tabs!
            return text

        # find the most likely indentation. to do that, find how many
        # lines have indentation that is a multiple of the candidate.
        indentation_candidates = {
            k: sum(1 if line_indentation % k == 0 else 0 for line_indentation in indentation_lengths)
            # assumption: the indentation is at least 2 spaces
            for k in range(2, max(indentation_lengths) + 1, 2)
        }

        # a bit hacky, but i'll pick which candidates are most likely
        # by checking if they are within sqrt of the highest candidate.
        most_likely_indentation, _ = max(indentation_candidates.items(), key=lambda x: max(sqrt(x[1]), x[0]))

        fn = partial(self._replace_leading_spaces, indentation=most_likely_indentation)
        return re.sub(r"^( +)", fn, text, flags=re.MULTILINE)

    def normalize(self, text: str) -> str:
        # 1. fix text
        text = cut_and_ftfy_text(text)

        # 2. remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 3. remove diacritics
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")

        # 5. replace the most likely indentation in space
        text = self._replace_tabs_with_spaces(text)

        # 6. escape spacing characters
        text = re.sub(r"\r?\n", "¶", text)  # end of paragraph marker for multiple newlines
        text = re.sub(r"\t", "↦", text)  # use ↦ for tabs
        text = text.strip()

        return text
