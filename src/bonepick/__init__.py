from bonepick.train_loop import (
    train_fasttext,
    train_model2vec,
)
from bonepick.eval_loop import eval_fasttext, eval_model2vec
from bonepick.data_loop import (
    balance_dataset,
    convert_to_fasttext,
    import_hf_dataset,
    normalize_dataset,
    transform_dataset,
)
from bonepick.cli import cli

__all__ = ["cli"]

cli.add_command(balance_dataset)
cli.add_command(convert_to_fasttext)
cli.add_command(eval_fasttext)
cli.add_command(eval_model2vec)
cli.add_command(import_hf_dataset)
cli.add_command(normalize_dataset)
cli.add_command(train_fasttext)
cli.add_command(train_model2vec)
cli.add_command(transform_dataset)
