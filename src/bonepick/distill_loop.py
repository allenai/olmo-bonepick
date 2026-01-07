from pathlib import Path
from typing import Literal

import click
import smart_open
from model2vec.distill import distill

from bonepick.cli import PathParamType, PCADimTypeParamType


@click.command()
@click.option("-m", "--model-name-or-path", type=str, required=True)
@click.option("-v", "--vocabulary-path", type=PathParamType(exists=True, is_file=True, optional=True), default=None)
@click.option("-o", "--output-dir", type=PathParamType(mkdir=True, is_dir=True), required=True)
@click.option("-d", "--pca-dims", type=PCADimTypeParamType(), default=256)
@click.option("-s", "--sif-coefficient", type=float, default=1e-4)
@click.option("-t", "--token-remove-pattern", type=str, default=r"\[unused\d+\]")
@click.option("-r", "--trust-remote-code", is_flag=True, default=False)
@click.option("-q", "--quantize-to", default="float16", type=click.Choice(["float16", "float32", "float64", "int8"]))
@click.option("-k", "--vocabulary-quantization", type=int, default=None)
@click.option("-p", "--pooling", default="mean", type=click.Choice(["mean", "last", "first", "pooler"]))
def distill_model(
    model_name_or_path: str,
    vocabulary_path: Path | None,
    output_dir: Path,
    pca_dims: int | None | float | Literal["auto"] = 256,
    sif_coefficient: float = 1e-4,
    token_remove_pattern: str = r"\[unused\d+\]",
    trust_remote_code: bool = False,
    quantize_to: str = "float16",
    vocabulary_quantization: int | None = None,
    pooling: str = "mean",
):
    click.echo("Starting model distillation...")

    # load vocabulary if provided
    if vocabulary_path is not None:
        click.echo(f"Loading vocabulary from {vocabulary_path}...")
        with smart_open.open(vocabulary_path, "rt", encoding="utf-8") as f: # pyright: ignore
            vocabulary = [line.strip() for line in f]
        click.echo(f"Vocabulary loaded successfully with {len(vocabulary)} tokens.")
    else:
        vocabulary = None

    # print distillation parameters
    click.echo(f"Distilling model {model_name_or_path}")
    click.echo(f"PCA dimensions: {pca_dims}")
    click.echo(f"SIF coefficient: {sif_coefficient}")
    click.echo(f"Token remove pattern: {token_remove_pattern}")
    click.echo(f"Trust remote code: {trust_remote_code}")
    click.echo(f"Quantize to: {quantize_to}")
    click.echo(f"Vocabulary quantization: {vocabulary_quantization}")
    click.echo(f"Pooling: {pooling}")

    # Distill a Sentence Transformer model
    m2v_model = distill(
        model_name=model_name_or_path,
        vocabulary=vocabulary,
        pca_dims=pca_dims,
        sif_coefficient=sif_coefficient,
        token_remove_pattern=token_remove_pattern,
        trust_remote_code=trust_remote_code,
        quantize_to=quantize_to,
        vocabulary_quantization=vocabulary_quantization,
        pooling=pooling,
    )
    click.echo(f"Model distilled successfully; saving to {output_dir}...")

    # Save the model
    m2v_model.save_pretrained(output_dir)
    click.echo("Distillation complete!")
