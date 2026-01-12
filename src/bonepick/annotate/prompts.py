import dataclasses as dt
import json
import re
from enum import Enum
from typing import (
    Generic,
    TypeVar,
    ClassVar,
    cast as typing_cast,
    TypedDict,
    Self,
)

from lazy_imports import try_import

from .annotate_utils import DataclassType

with try_import() as extra_dependencies:
    from bonepick.annotate.pydantic_utils import dataclass_to_json_schema


T = TypeVar("T")


class TurnRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TurnPosition(Enum):
    FIRST = "first"
    LAST = "last"


class TurnDict(TypedDict):
    role: TurnRole
    content: str


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]

THINK_START = "<think>"
THINK_END = "</think>"


class ContentOrder(Enum):
    INSTRUCTIONS_CONTENT = "instructions_content"
    CONTENT_INSTRUCTIONS = "content_instructions"


@dt.dataclass(frozen=True)
class BasePrompt(Generic[T]):
    name: str = dt.field(default_factory=str)
    preamble: str = dt.field(default_factory=str)
    instructions: str = dt.field(default_factory=str)
    role_to_annotate: TurnRole = dt.field(default=TurnRole.USER)
    turn_to_annotate: TurnPosition = dt.field(default=TurnPosition.LAST)
    output_type: type[DataclassType] | None = dt.field(default=None)
    content_order: ContentOrder = dt.field(default=ContentOrder.CONTENT_INSTRUCTIONS)

    _registry: ClassVar[dict[str, type[Self]]]

    def __post_init__(self):
        assert self.name.strip(), "Prompt name cannot be empty"
        assert self.instructions.strip(), "Prompt instructions cannot be empty"
        assert self.role_to_annotate in TurnRole, "Invalid role to annotate"
        assert self.turn_to_annotate in TurnPosition, "Invalid turn to annotate"

        # check if imports are fine
        extra_dependencies.check()

    @property
    def null(self) -> T:
        return typing_cast(T, "")

    def postprocess(self, text: str) -> T:
        return typing_cast(T, text.strip("'").strip('"').strip())

    def sanitize_text(self, text: str) -> str:
        for special_token in SPECIAL_TOKENS:
            if special_token in text:
                replacement = " ".join(re.split(r"\b", special_token))
                text = text.replace(special_token, replacement)
        return text.strip()

    def clean_turn(self, turn: TurnDict) -> TurnDict:
        if turn["role"] == TurnRole.ASSISTANT.value:
            start_loc = turn["content"].find(THINK_START)
            end_loc = turn["content"].rfind(THINK_END)

            if start_loc >= 0 and end_loc >= 0:
                turn["content"] = (
                    turn["content"][:start_loc] + turn["content"][end_loc + len(THINK_END) :]
                ).strip()
            elif start_loc >= 0:
                turn["content"] = turn["content"].replace(THINK_START, "")
            elif end_loc >= 0:
                turn["content"] = turn["content"].replace(THINK_END, "")

        # remove special tokens
        turn["content"] = self.sanitize_text(turn["content"])

        # removes excessive newlines
        turn["content"] = re.sub(r"\n{3,}", "\n\n", turn["content"]).strip()

        return turn

    def filter(self, conversation: list[TurnDict]) -> list[TurnDict]:
        select_fn = max if self.turn_to_annotate == TurnPosition.LAST else min
        last_turn = select_fn([i for i, c in enumerate(conversation) if c["role"] == self.role_to_annotate.value])
        return [self.clean_turn(turn) for i, turn in enumerate(conversation) if i <= last_turn]

    @property
    def schema(self) -> dict | None:
        if self.output_type is None:
            return None
        return dataclass_to_json_schema(self.output_type)

    def parse(self, response_text: str) -> dict | str:
        if self.output_type is None:
            return response_text
        return json.loads(response_text)

    def subset(self, batch: dict[str, list]) -> list[bool]:
        """Use this function to subset a dataset based on the content of each example."""
        if len(batch) == 0:
            raise ValueError("Batch is empty")
        batch_key = list(batch.keys())[0]
        return [True] * len(batch[batch_key])

    @classmethod
    def register(cls, prompt: type[Self]) -> type[Self]:
        assert hasattr(cls, "_registry"), "BasePrompt must be subclassed"
        cls._registry[prompt.name] = prompt
        return prompt

    @classmethod
    def get(cls, name: str) -> Self:
        assert hasattr(cls, "_registry"), "BasePrompt must be subclassed"
        if name not in cls._registry:
            raise ValueError(f"Prompt '{name}' not found")
        return cls._registry[name]()

    @classmethod
    def prompts(cls) -> list[str]:
        return [prompt.name for prompt in cls._registry.values()]

    def format_conversation(self, messages: list[TurnDict], max_text_length: int | None = None) -> str:
        messages = self.filter(messages)
        formatted_messages = ""
        for turn in messages:
            formatted_messages += f"ROLE:{turn['role']}\nCONTENT:{self.clean_turn(turn)['content']}\n\n\n"

        if max_text_length is not None and len(formatted_messages) > max_text_length:
            if self.turn_to_annotate == TurnPosition.LAST:
                formatted_messages = formatted_messages[:max_text_length]
            else:
                formatted_messages = formatted_messages[-max_text_length:]

        return f"CONVERSATION:\n{formatted_messages.strip()}"

    def format_text(self, text: str, max_text_length: int | None = None) -> str:
        if max_text_length is not None and len(text) > max_text_length:
            text = text[:max_text_length]
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return f"TEXT:\n{text}"

    def format_instructions(self) -> str:
        return f"INSTRUCTIONS:\n{self.instructions.strip()}"

    def format_preamble(self) -> str:
        return self.preamble.strip()

    def separator(self) -> str:
        return "\n\n\n"

    def apply(
        self,
        conversation_or_text: list[TurnDict] | str | None = None,
        max_text_length: int | None = None,
    ) -> str:
        if conversation_or_text is None:
            # this is for the case of system prompts
            return self.instructions.strip()

        # format conversation or text
        content = (
            self.format_conversation(messages=conversation_or_text, max_text_length=max_text_length)
            if isinstance(conversation_or_text, list)
            else self.format_text(text=conversation_or_text, max_text_length=max_text_length)
        ).strip()

        # sanitize and concatenate content and instructions
        sanitized_content = self.sanitize_text(content)
        formatted_content = self.separator().join([sanitized_content, self.format_instructions()])

        if (preamble_text := self.format_preamble()) and preamble_text.strip():
            formatted_content = self.separator().join([preamble_text.strip(), formatted_content.strip()])

        return formatted_content.strip()


@dt.dataclass(frozen=True)
class BaseSystemPrompt(BasePrompt, Generic[T]):
    role_to_annotate: TurnRole = TurnRole.SYSTEM
    turn_to_annotate: TurnPosition = TurnPosition.FIRST
    _registry: ClassVar[dict[str, type[Self]]] = {}


@dt.dataclass(frozen=True)
class BaseAnnotationPrompt(BasePrompt, Generic[T]):
    _registry: ClassVar[dict[str, type[Self]]] = {}
