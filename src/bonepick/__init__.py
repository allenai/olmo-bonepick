import os

# this before any other import (specifically before datasets)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import multiprocessing

import click
import smart_open.compression
import torch

from bonepick.annotate import annotate_dataset, annotation_agreement, label_distribution, list_prompts
from bonepick.cli import cli  # noqa: E402
from bonepick.data import (
    balance_dataset,
    convert_to_fasttext,
    count_tokens,
    import_hf_dataset,
    normalize_dataset,
    reshard_dataset,
    sample_dataset,
    transform_dataset,
)
from bonepick.evals import eval_calibration, train_calibration
from bonepick.fasttext import eval_fasttext, infer_fasttext, train_fasttext
from bonepick.logger import init_logger  # noqa: E402
from bonepick.model2vec import distill_model2vec, eval_model2vec, train_model2vec
from bonepick.version import __version__

__all__ = ["cli", "__version__"]

# set start method for multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# suppresses following warning:
#   You are using a CUDA device ('NVIDIA GB10') that has Tensor Cores. To properly utilize them, you should set
#   `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance.
#   For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")

# initialize logger
init_logger()

# add additional extension for ZST compression
smart_open.compression.register_compressor(".zstd", smart_open.compression._handle_zstd)

cli.add_command(balance_dataset)
cli.add_command(convert_to_fasttext)
cli.add_command(count_tokens)
cli.add_command(distill_model2vec)
cli.add_command(eval_fasttext)
cli.add_command(eval_model2vec)
cli.add_command(import_hf_dataset)
cli.add_command(infer_fasttext)
cli.add_command(normalize_dataset)
cli.add_command(reshard_dataset)
cli.add_command(sample_dataset)
cli.add_command(train_fasttext)
cli.add_command(train_model2vec)
cli.add_command(transform_dataset)
cli.add_command(annotate_dataset)
cli.add_command(list_prompts)
cli.add_command(annotation_agreement)
cli.add_command(label_distribution)
cli.add_command(eval_calibration)
cli.add_command(train_calibration)


@cli.command()
def version():
    """Print the version of the package and exit"""
    click.echo(f"{__package__} {__version__}")
