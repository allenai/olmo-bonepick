import hashlib
import shutil
from pathlib import Path

import click
import smart_open


def check_fasttext_binary() -> Path:
    """Check if fasttext binary is available and return its path."""
    click.echo("Checking for fasttext binary in PATH...")
    fasttext_path = shutil.which("fasttext")
    if fasttext_path is None:
        raise click.ClickException(
            "fasttext binary not found in PATH. Please install fasttext: https://fasttext.cc/docs/en/support.html"
        )
    click.echo(f"Found fasttext binary at: {fasttext_path}")
    return Path(fasttext_path)


def fasttext_dataset_signature(fasttext_file: Path) -> str:
    assert fasttext_file.exists(), f"Fasttext file {fasttext_file} does not exist"
    assert fasttext_file.is_file(), f"Fasttext file {fasttext_file} is not a file"

    h = hashlib.sha256()
    with smart_open.open(fasttext_file, "rb") as f:  # pyright: ignore
        for line in f:
            h.update(line)
    return h.hexdigest()
