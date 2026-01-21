from functools import reduce
from typing import Any, Callable, Protocol, TypeVar

import click
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
    ) -> Any: ...


T = TypeVar("T", bound=FieldOrExpressionCommandProtocol)


def generate_calibration_jq_expression(
    weights: dict[str, float],
    bias: float,
    model_type: str,
    source_expression: str,
) -> str:
    """Generate a jq expression to compute a weighted score from a dict.

    The output expression pipes source_expression into the weighted sum,
    e.g.: source_expression | ((."key1" * w1) + (."key2" * w2) + bias)

    Args:
        weights: Component weights
        bias: Bias term
        model_type: "linear" or "log-linear"
        source_expression: JQ expression that extracts the source dict

    Returns:
        JQ expression string that computes the weighted score
    """
    # Build the sum expression using . to reference the piped input
    terms = []
    for name, weight in sorted(weights.items()):
        if weight >= 0:
            terms.append(f'(."{name}" * {weight:.6f})')
        else:
            terms.append(f'(."{name}" * ({weight:.6f}))')

    sum_expr = " + ".join(terms)
    if bias >= 0:
        sum_expr = f"({sum_expr} + {bias:.6f})"
    else:
        sum_expr = f"({sum_expr} + ({bias:.6f}))"

    if model_type == "linear":
        # Linear: unbounded weighted sum
        return f"{source_expression} | {sum_expr}"
    else:  # log-linear
        # Apply sigmoid: 1 / (1 + exp(-x)), pipe source_expression into the computation
        return f"{source_expression} | (1 / (1 + ((-1 * {sum_expr}) | exp)))"


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
