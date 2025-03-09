from typing import Union, TypedDict, Literal


class Message(TypedDict):
    content: str
    role: Literal["system", "user", "assistant"]

ChatFormatRequest = Union[str, list[str], list[Message], None]
