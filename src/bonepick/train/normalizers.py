import re
from typing import Callable, Generic, TypeVar
import unicodedata
from plsfix import fix_text

from tokenizers import Tokenizer


__all__ = [
    "register_normalizer",
    "get_normalizer",
    "list_normalizers",
]


T = TypeVar("T", bound="BaseRowNormalizer")


NORMALIZER_REGISTRY: dict[str, type["BaseRowNormalizer"]] = {}


def fix_and_cut_text(text: str, max_length: int = 2**20) -> str:
    if len(text) > max_length:
        next_newline = text.find("\n", max_length)
        text = text[:max_length] if next_newline == -1 else text[:next_newline]
    try:
        return fix_text(text)
    except Exception:
        if max_length < 0:
            return text
        return fix_and_cut_text(text, max_length=max_length // 2)


def register_normalizer(name: str) -> Callable[[type[T]], type[T]]:
    def decorator(normalizer_cls: type[T]) -> type[T]:
        if (
            name in NORMALIZER_REGISTRY
            and normalizer_cls is not NORMALIZER_REGISTRY[name]
        ):
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
        return fix_and_cut_text(text)


@register_normalizer("tokenizer")
class TokenizerNormalizer(BaseRowNormalizer):
    def __init__(self):
        self.tokenizer = Tokenizer.from_pretrained("allenai/dolma2-tokenizer")

    def normalize(self, text: str) -> str:
        cleaned_text = fix_and_cut_text(text)
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
        text = "".join(
            c
            for c in unicodedata.normalize("NFKD", text)
            if unicodedata.category(c) != "Mn"
        )

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
        self.tokenizer = Tokenizer.from_pretrained("allenai/Ultra-FineWeb-tokenizer")

    def normalize(self, text: str) -> str:
        # 1. fix text
        text = fix_and_cut_text(text)

        # 2. remove multiple newlines
        text = self.remove_extra_newlines_re.sub("\n\n", text)

        # 3. lower the text
        text = text.lower()

        # 4. remove diacritics
        text = "".join(
            c
            for c in unicodedata.normalize("NFKD", text)
            if unicodedata.category(c) != "Mn"
        )

        # 5. add spacing around special characters
        text = self.space_before_special_re.sub(" \\1 ", text)

        # 6. first removal of extra whitespace
        text = self.whitespace_re.sub(" ", text)

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
        text = fix_and_cut_text(text)

        # 2. remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 3. lower the content
        text = text.lower()

        # 4. remove diacritics
        text = "".join(
            c
            for c in unicodedata.normalize("NFKD", text)
            if unicodedata.category(c) != "Mn"
        )

        # 5. escape spacing characters
        text = re.sub(
            r"\r?\n\r?\n", "¶", text
        )  # end of paragraph marker for multiple newlines
        text = re.sub(r"\r?\n", "·", text)  # just cdot for single newlines
        text = re.sub(r"\t", "↦", text)  # use ↦ for tabs
        text = re.sub(r"\s+", " ", text)  # remove extra whitespace
        text = text.strip()

        return text
