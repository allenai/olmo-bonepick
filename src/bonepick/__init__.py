import os

# this before any other import (specifically before datasets)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"


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
from bonepick.cli import cli

__all__ = ["cli"]

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
