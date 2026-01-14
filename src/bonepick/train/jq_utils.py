import click
from typing import Callable, Any, Protocol, TypeVar
from functools import reduce

import jq


def compile_jq(jq_expr: str) -> Callable[[dict], Any]:
    if not jq_expr.strip():

        def identity(x: dict) -> dict:
            assert isinstance(x, dict), f"Expected dict, got {type(x)}"
            return x

        return identity

    compiled_jq = jq.compile(jq_expr)

    def transform(x: dict, _compiled_jq=compiled_jq) -> dict:
        assert isinstance(x, dict), f"Expected dict, got {type(x)}"
        output = _compiled_jq.input_value(x).first()
        assert output is not None, "Expected output, got None"
        return output

    return transform


def field_or_expression(field: str | None = None, expression: str | None = None) -> str:
    if field is not None:
        msg = (
            "[bold red]WARNING:[/bold red] [red]-t/--text-field[/red] is deprecated, "
            "use [red]-tt/--text-expression[/red] instead."
        )
        click.echo(msg, err=True, color=True)
        return f".{field}"

    if expression is None:
        raise ValueError("Either field or expression must be provided")

    return expression


class FieldOrExpressionCommandProtocol(Protocol):

    def __call__(
        self,
        text_field: str | None,
        label_field: str | None,
        text_expression: str,
        label_expression: str,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ...


T = TypeVar("T", bound=FieldOrExpressionCommandProtocol)


def add_field_or_expression_command_options(
    command_fn: FieldOrExpressionCommandProtocol,
) -> FieldOrExpressionCommandProtocol:
    click_decorators = [
        click.option(
            "-t",
            "--text-field",
            type=str,
            default=None,
            help="Field in dataset to use as text",
        ),
        click.option(
            "-l",
            "--label-field",
            type=str,
            default=None,
            help="Field in dataset to use as label",
        ),
        click.option(
            "-tt",
            "--text-expression",
            type=str,
            default=".text",
            help="expression to extract text from dataset",
        ),
        click.option(
            "-ll",
            "--label-expression",
            type=str,
            default=".score",
            help="expression to extract label from dataset",
        ),
    ]
    return reduce(lambda f, decorator: decorator(f), click_decorators, command_fn)
