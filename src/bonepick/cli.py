import click
from pathlib import Path


@click.group()
def cli():
    """
    Bonepick: Train efficient text quality classifiers using Model2Vec and FastText.

    A complete pipeline for building fast, CPU-friendly text classifiers from HuggingFace
    datasets through data preparation (import, transform, balance, normalize) to training
    (Model2Vec with standard or contrastive learning, FastText) and evaluation.
    """
    pass


class FloatOrIntParamType(click.ParamType):
    name = "float | int"

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise self.fail(f"{value!r} is not a valid float or int", param, ctx)

        if isinstance(value, float) and value.is_integer():
            return int(value)

        if not isinstance(value, (float, int)):
            raise self.fail(f"{value!r} is not a valid float or int", param, ctx)

        return value


class PathParamType(click.ParamType):
    name = "path"

    def __init__(
        self,
        exists: bool = False,
        mkdir: bool = False,
        is_dir: bool = False,
        is_file: bool = False,
        optional: bool = False,
    ):
        self.exists = exists
        self.mkdir = mkdir
        self.is_dir = is_dir
        self.is_file = is_file
        self.optional = optional

    def convert(self, value, param, ctx):
        if self.optional and value is None:
            return None

        if isinstance(value, Path):
            path = value
        elif isinstance(value, str):
            path = Path(value)
        else:
            raise self.fail(f"{value!r} is not a valid path", param, ctx)

        if self.exists and not path.exists():
            raise self.fail(f"{path!r} does not exist", param, ctx)

        if self.mkdir:
            path.mkdir(parents=True, exist_ok=True)

        if self.is_dir and not path.is_dir():
            raise self.fail(f"{path!r} is not a directory", param, ctx)

        if self.is_file and not path.is_file():
            raise self.fail(f"{path!r} is not a file", param, ctx)

        return path


class PCADimTypeParamType(FloatOrIntParamType):
    name = "pca-dims"

    def convert(self, value, param, ctx):
        if value is None:
            return None

        if isinstance(value, str) and value.lower() == "auto":
            return "auto"

        return super().convert(value, param, ctx)


class ByteSizeParamType(click.ParamType):
    name = "bytesize"

    def convert(self, value, param, ctx):
        if isinstance(value, int):
            return value

        if not isinstance(value, str):
            raise self.fail(f"{value!r} is not a valid byte size", param, ctx)

        # Parse string like "1GB", "500MB", "1.5KB", etc.
        value = value.strip().upper()

        # Extract number and unit
        import re

        match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$", value)
        if not match:
            raise self.fail(
                f"{value!r} is not a valid byte size. Use format like '1GB', '500MB', '1.5KB'",
                param,
                ctx,
            )

        num_str, unit = match.groups()
        num = float(num_str)

        # Convert to bytes
        multipliers = {
            "B": 1,
            "": 1,
            "KB": 1024,
            "K": 1024,
            "MB": 1024**2,
            "M": 1024**2,
            "GB": 1024**3,
            "G": 1024**3,
            "TB": 1024**4,
            "T": 1024**4,
        }

        if unit not in multipliers:
            raise self.fail(f"Unknown unit {unit!r}", param, ctx)

        return int(num * multipliers[unit])
