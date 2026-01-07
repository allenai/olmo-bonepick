import click
from pathlib import Path


@click.group()
def cli():
    """Train fast CPU classifiers."""
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

    def __init__(self, exists: bool = False, mkdir: bool = False, is_dir: bool = False, is_file: bool = False):
        self.exists = exists
        self.mkdir = mkdir
        self.is_dir = is_dir
        self.is_file = is_file

    def convert(self, value, param, ctx):
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
