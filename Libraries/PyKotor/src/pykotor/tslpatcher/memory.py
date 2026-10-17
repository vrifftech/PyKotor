from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utility.system.path import PureWindowsPath


class _Memory2DA(dict[int, "str | PureWindowsPath"]):
    """Installer slots are assigned or overwritten, never removed."""

    def __init__(self):
        super().__init__()
        self.highest_slot: int = 0

    def __setitem__(self, token_id: int, value: str | PureWindowsPath) -> None:
        super().__setitem__(token_id, value)
        self.highest_slot = max(self.highest_slot, token_id)


class PatcherMemory:
    def __init__(self):
        self.memory_2da = _Memory2DA()  # 2DAMemory# (token) -> str or field path
        self.memory_str: dict[int, int] = {}  # StrRef# (token) -> dialog.tlk index

    def __repr__(self):
        memory_2da_repr: dict[int, str] = {key: repr(value) for key, value in self.memory_2da.items()}
        memory_str_repr: dict[int, str] = {key: repr(value) for key, value in self.memory_str.items()}
        return f"PatcherMemory(memory_2da={memory_2da_repr}, memory_str={memory_str_repr})"


class TokenUsage(ABC):
    @abstractmethod
    def value(self, memory: PatcherMemory) -> str: ...


class NoTokenUsage(TokenUsage):
    def __init__(self, stored: str | int):
        self.stored = str(stored)

    def value(self, memory: PatcherMemory) -> str:
        return self.stored


class TokenUsage2DA(TokenUsage):
    def __init__(self, token_id: int):
        self.token_id: int = token_id

    def value(self, memory: PatcherMemory) -> str | PureWindowsPath:  # type: ignore[override]
        memory_val: str | PureWindowsPath | None = memory.memory_2da.get(self.token_id, None)
        if memory_val is None:
            msg = f"2DAMEMORY{self.token_id} was not defined before use"
            raise KeyError(msg)
        return memory_val


class TokenUsageTLK(TokenUsage):
    def __init__(self, token_id: int):
        self.token_id: int = token_id

    def value(self, memory: PatcherMemory) -> str:
        memory_val: int | None = memory.memory_str.get(self.token_id, None)
        if memory_val is None:
            msg = f"StrRef{self.token_id} was not defined before use"
            raise KeyError(msg)
        return str(memory_val)
