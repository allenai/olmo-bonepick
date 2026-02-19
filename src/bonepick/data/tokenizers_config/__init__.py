from pathlib import Path

_TOKENIZERS_DIR = Path(__file__).parent

DOLMA2_TOKENIZER_PATH = str(_TOKENIZERS_DIR / "dolma2_tokenizer.json")
ULTRA_FINEWEB_TOKENIZER_PATH = str(_TOKENIZERS_DIR / "ultra_fineweb_tokenizer.json")

assert Path(DOLMA2_TOKENIZER_PATH).exists(), (
    f"File {DOLMA2_TOKENIZER_PATH} does not exist; please report this as an issue"
)
assert Path(ULTRA_FINEWEB_TOKENIZER_PATH).exists(), (
    f"File {ULTRA_FINEWEB_TOKENIZER_PATH} does not exist; please report this as an issue"
)

__all__ = ["DOLMA2_TOKENIZER_PATH", "ULTRA_FINEWEB_TOKENIZER_PATH"]
