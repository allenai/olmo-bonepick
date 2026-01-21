import re
import unicodedata
from itertools import batched
from typing import Callable, Generic, TypeVar

from anyascii import anyascii
from plsfix import fix_text
from tokenizers import Tokenizer

from bonepick.data.indentation import convert_spaces_to_tabs

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


@register_normalizer("hyperfine")
class HyperFineNormalizer(BaseRowNormalizer):
    def __init__(self):
        self.remove_extra_newlines_re = re.compile(r"\n{3,}")
        self.segment_symbols_re = re.compile(r"([\t\r\n]+)")
        self.replace_spaces_re = re.compile(r"(\s+)")
        self.tokenizer: Tokenizer = Tokenizer.from_pretrained(
            "allenai/dolma2-tokenizer"
        )  # equivalent to cl100k_base

    def normalize(self, text: str) -> str:
        # 1. fix text with faster version of ftfy
        text = cut_and_ftfy_text(text)

        # 2. remove multiple newlines
        text = self.remove_extra_newlines_re.sub("\n\n", text)

        # 3. lower the text
        text = text.lower()

        # 4. remove diacritics
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")

        # 5. split into segments
        segments, spaces = zip(
            # we need the + [""] to handle the fact that we always have one fewer element in the
            # \n\t\r block, even when a string ends with e.g. \n.
            # for example `self.segment_symbols_re.split('\n')` returns ['', '\n', '']
            *batched(self.segment_symbols_re.split(text) + [""], 2)
        )

        # 6. this replaces multiple spaces with a single one
        escaped_segments = [self.replace_spaces_re.sub(" ", str(segment)).strip() for segment in segments]

        # 7. this encodes each segment with a tokenizer
        #    (can't use encode_batch_fast because it won't return tokens)
        encoded_segments = self.tokenizer.encode_batch(escaped_segments, add_special_tokens=False)

        # 8. this escapes \n, \t, and \r to \\n, \\t, and \\r
        #    the " ".join puts spaces between the escaped characters
        escaped_spaces = [" ".join(str(space).encode("unicode_escape").decode("ascii")) for space in spaces]

        # 9. put the string back together (strip ensures no stray spaces)
        text = " ".join(
            (" ".join(encoded_segment.tokens) + " " + escaped_space)
            for encoded_segment, escaped_space in zip(encoded_segments, escaped_spaces)
        ).strip()

        return text


@register_normalizer("hyperfine-code")
class HyperFineCodeNormalizer(HyperFineNormalizer):
    def normalize(self, text: str) -> str:
        # 1. fix text with faster version of ftfy
        text = cut_and_ftfy_text(text)

        # 2. remove multiple newlines
        text = self.remove_extra_newlines_re.sub("\n\n", text)

        # 3. remove diacritics
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")

        # 4. if text has space indentation, we convert it to tabs
        text = convert_spaces_to_tabs(text)

        # 5. split into segments
        segments, spaces = zip(
            # we need the + [""] to handle the fact that we always have one fewer element in the
            # \n\t\r block, even when a string ends with e.g. \n.
            # for example `self.segment_symbols_re.split('\n')` returns ['', '\n', '']
            *batched(self.segment_symbols_re.split(text) + [""], 2)
        )

        # 6. this encodes each segment with a tokenizer
        #    (can't use encode_batch_fast because it won't return tokens)
        encoded_segments = self.tokenizer.encode_batch(segments, add_special_tokens=False)

        # 7. this escapes \n, \t, and \r to \\n, \\t, and \\r
        #    the " ".join puts spaces between the escaped characters
        escaped_spaces = [" ".join(str(space).encode("unicode_escape").decode("ascii")) for space in spaces]

        # 8. put the string back together (strip ensures no stray spaces)
        text = " ".join(
            (" ".join(encoded_segment.tokens) + " " + escaped_space)
            for encoded_segment, escaped_space in zip(encoded_segments, escaped_spaces)
        ).strip()

        return text


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
        self.encode_newlines_re = re.compile(r"\r?\n")
        self.encode_tabs_re = re.compile(r"\t")

    def normalize(self, text: str) -> str:
        # 1. fix text
        text = cut_and_ftfy_text(text)

        # 2. remove multiple newlines
        text = self.remove_extra_newlines_re.sub("\n\n", text)

        # 3. remove diacritics
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn")

        # 4. replace the most likely indentation in space
        text = convert_spaces_to_tabs(text)

        # 5. escape spacing characters
        text = self.encode_newlines_re.sub("¶", text)  # end of paragraph marker for multiple newlines
        text = self.encode_tabs_re.sub("↦", text)  # use ↦ for tabs
        text = text.strip()

        return text
