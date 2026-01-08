from typing import Protocol, ClassVar, Any


class DataclassType(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]
