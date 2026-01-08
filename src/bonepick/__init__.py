import os
import sys
import io

# this before any other import (specifically before datasets)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import multiprocessing

import click

from bonepick.version import __version__

# Suppress "Warning: plyvel not installed" from lm_deluge during import
_original_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from bonepick.train import (
        balance_dataset,
        convert_to_fasttext,
        distill_model,
        eval_fasttext,
        eval_model2vec,
        import_hf_dataset,
        normalize_dataset,
        train_fasttext,
        train_model2vec,
        transform_dataset,
    )
finally:
    sys.stdout = _original_stdout

from bonepick.cli import cli    # noqa: E402
from bonepick.logger import init_logger # noqa: E402

__all__ = ["cli", "__version__"]

# set start method for multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# initialize logger
init_logger()

cli.add_command(balance_dataset)
cli.add_command(convert_to_fasttext)
cli.add_command(distill_model)
cli.add_command(eval_fasttext)
cli.add_command(eval_model2vec)
cli.add_command(import_hf_dataset)
cli.add_command(normalize_dataset)
cli.add_command(train_fasttext)
cli.add_command(train_model2vec)
cli.add_command(transform_dataset)


@cli.command()
def version():
    """Print the version of the package and exit"""
    click.echo(f"{__package__} {__version__}")
