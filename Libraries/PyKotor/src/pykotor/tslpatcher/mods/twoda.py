from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING

from pykotor.common.misc import CaseInsensitiveDict
from pykotor.resource.formats.twoda import bytes_2da, read_2da
from pykotor.tslpatcher.mods.template import PatcherModifications
from utility.error_handling import universal_simplify_exception
from utility.logger_util import RobustRootLogger
from utility.system.path import PureWindowsPath

if TYPE_CHECKING:
    from typing_extensions import Literal

    from pykotor.common.misc import Game
    from pykotor.resource.formats.twoda import TwoDA, TwoDARow
    from pykotor.resource.type import SOURCE_TYPES
    from pykotor.tslpatcher.logger import PatchLogger
    from pykotor.tslpatcher.memory import PatcherMemory


class CriticalError(Exception): ...


class WarningError(Exception): ...


class TargetType(IntEnum):
    ROW_INDEX = 0
    ROW_LABEL = 1
    LABEL_COLUMN = 2


class Target:
    def __init__(self, target_type: TargetType, value: str | int | RowValue2DAMemory | RowValueTLKMemory):
        self.target_type: TargetType = target_type
        self.value: str | int | RowValueTLKMemory | RowValue2DAMemory = value

        if target_type == TargetType.ROW_INDEX and isinstance(value, str):
            msg = "Target value must be int if type is row index."
            raise ValueError(msg)

    def __repr__(self):
        return f"{self.__class__.__name__}(target_type={self.target_type.__class__.__name__}.{self.target_type.name}, value={self.value!r})"

    def search(
        self,
        twoda: TwoDA,
        memory: PatcherMemory,
    ) -> TwoDARow | None:
        """Searches a TwoDA for a row matching the target.

        Args:
        ----
            twoda: TwoDA - The TwoDA to search
            target_type: TargetType - The type of target to search for
        Returns:
            TwoDARow | None - The matching row if found, else None
        Processing Logic:
        ----------------
            - Checks target_type and searches twoda accordingly
            - For row index, gets row directly
            - For row label, finds row by label
            - For label column, checks for label column, then iterates rows to find match
            - Returns matching row or None.
        """
        if isinstance(self.value, (RowValueTLKMemory, RowValue2DAMemory)):
            value = self.value.value(memory, twoda, None)
        else:
            value = self.value
        source_row: TwoDARow | None = None
        if self.target_type == TargetType.ROW_INDEX:
            row_index = int(value)
            if 0 <= row_index < twoda.get_height():
                source_row = twoda.get_row(row_index)
        elif self.target_type == TargetType.ROW_LABEL:
            source_row = twoda.find_row(str(value))
        elif self.target_type == TargetType.LABEL_COLUMN:
            label_header = next((header for header in twoda.get_headers() if header.lower() == "label"), None)
            if label_header is None:
                msg = f"'label' could not be found in the twoda's headers: ({self.target_type.name}, {value})"
                raise WarningError(msg)
            if value not in twoda.get_column(label_header):
                msg = f"The value '{value}' could not be found in the twoda's columns"
                raise WarningError(msg)
            for row in twoda:
                if row.get_string(label_header) == value:
                    source_row = row

        return source_row


# region Value Returners
class RowValue(ABC):
    @abstractmethod
    def value(self, memory: PatcherMemory, twoda: TwoDA, row: TwoDARow | None) -> str: ...


class RowValueConstant(RowValue):
    def __init__(self, string: str):
        self.string: str = string

    def __repr__(self):
        return f"{self.__class__.__name__}(string='{self.string}')"

    def value(self, memory: PatcherMemory, twoda: TwoDA, row: TwoDARow | None) -> str:
        return self.string


class RowValue2DAMemory(RowValue):
    def __init__(self, token_id: int):
        self.token_id: int = token_id

    def __repr__(self):
        return f"{self.__class__.__name__}(token_id={self.token_id})"

    def value(self, memory: PatcherMemory, twoda: TwoDA, row: TwoDARow | None) -> str:
        memory_val: str | PureWindowsPath | None = memory.memory_2da.get(self.token_id)
        if memory_val is None:
            msg = f"2DAMEMORY{self.token_id} was not defined before use."
            raise KeyError(msg)
        if isinstance(memory_val, PureWindowsPath):
            msg = f"!FieldPath cannot be used in 2DAList patches, got '{memory_val}'"
            raise TypeError(msg)
        return memory_val


class RowValueTLKMemory(RowValue):
    def __init__(self, token_id: int):
        self.token_id: int = token_id

    def __repr__(self):
        return f"{self.__class__.__name__}(token_id={self.token_id})"

    def value(self, memory: PatcherMemory, twoda: TwoDA, row: TwoDARow | None) -> str:
        memory_val: int | None = memory.memory_str.get(self.token_id)
        if memory_val is None:
            msg = f"StrRef{self.token_id} was not defined before use."
            raise KeyError(msg)
        return str(memory_val)


class RowValueHigh(RowValue):
    """
    Attributes:
    ----------
    column: Column to get the max integer from. If None it takes it from the Row Label.
    """  # noqa: D212, D415

    def __init__(self, column: str | None):
        self.column: str | None = column

    def __repr__(self):
        return f"{self.__class__.__name__}(column='{self.column}')"

    def value(self, memory: PatcherMemory, twoda: TwoDA, row: TwoDARow | None) -> str:
        """Returns the maximum value in a column or overall label.

        Args:
        ----
            memory: PatcherMemory object
            twoda: TwoDA object
            row: TwoDARow object or None

        Returns:
        -------
            str: String representation of maximum value

        Processing Logic:
        ----------------
            - If column is not None, return maximum value in that column
            - Else return overall maximum label value.
        """
        return str(twoda.label_max()) if self.column is None else str(twoda.column_max(self.column))


class RowValueRowIndex(RowValue):
    def __init__(self): ...

    def value(self, memory: PatcherMemory, twoda: TwoDA, row: TwoDARow | None) -> str:
        return "" if row is None else str(twoda.row_index(row))


class RowValueRowLabel(RowValue):
    def __init__(self): ...

    def value(self, memory: PatcherMemory, twoda: TwoDA, row: TwoDARow | None) -> str:
        return "" if row is None else row.label()


class RowValueRowCell(RowValue):
    def __init__(self, column: str):
        self.column: str = column

    def __repr__(self):
        return f"{self.__class__.__name__}(column='{self.column}')"

    def value(self, memory: PatcherMemory, twoda: TwoDA, row: TwoDARow | None) -> str:
        return "" if row is None else row.get_string(self.column)


# endregion


# region Modify 2DA
class Modify2DA(ABC):
    @abstractmethod
    def __init__(self): ...

    @staticmethod
    def _normalize_entries(entries: list[tuple[str, str]] | None) -> list[tuple[str, str]]:
        return [] if entries is None else [(str(key), "" if value is None else str(value)) for key, value in entries]

    @staticmethod
    def _is_unsigned_decimal(value: str) -> bool:
        return bool(value) and value.isascii() and value.isdigit()

    @staticmethod
    def _row_value_to_raw(value: RowValue | str | int) -> str:
        if isinstance(value, RowValueConstant):
            return value.string
        if isinstance(value, RowValue2DAMemory):
            return f"2DAMEMORY{value.token_id}"
        if isinstance(value, RowValueTLKMemory):
            return f"StrRef{value.token_id}"
        if isinstance(value, RowValueHigh):
            return "high()"
        if isinstance(value, RowValueRowIndex):
            return "RowIndex"
        if isinstance(value, RowValueRowLabel):
            return "RowLabel"
        if isinstance(value, RowValueRowCell):
            return value.column
        return str(value)

    @classmethod
    def _target_to_entry(cls, target: Target) -> tuple[str, str]:
        key = {
            TargetType.ROW_INDEX: "RowIndex",
            TargetType.ROW_LABEL: "RowLabel",
            TargetType.LABEL_COLUMN: "LabelIndex",
        }[target.target_type]
        return key, cls._row_value_to_raw(target.value)

    @staticmethod
    def _find_header(twoda: TwoDA, header: str) -> str | None:
        lowered = header.lower()
        return next((existing for existing in twoda.get_headers() if existing.lower() == lowered), None)

    @staticmethod
    def _find_row_index(twoda: TwoDA, row_label: str) -> int | None:
        lowered = row_label.lower()
        return next((index for index, label in enumerate(twoda.get_labels()) if label.lower() == lowered), None)

    @classmethod
    def _resolve_2da_memory(cls, value: str, memory: PatcherMemory) -> str:
        if not value.startswith("2DAMEMORY"):
            return value

        suffix = value[9:]
        slots = memory.memory_2da
        if not slots:
            return value

        highest_slot = max((token_id for token_id in slots if token_id > 0), default=0)
        if cls._is_unsigned_decimal(suffix):
            token_id = int(suffix)
            if token_id < 1:
                token_id = 1
            elif token_id > highest_slot:
                token_id = 1
        else:
            token_id = 1

        resolved = slots.get(token_id, "")
        if isinstance(resolved, PureWindowsPath):
            msg = f"!FieldPath cannot be used in 2DAList patches, got '{resolved}'"
            raise TypeError(msg)
        return resolved

    @classmethod
    def _resolve_strref(cls, value: str, memory: PatcherMemory) -> str:
        if len(value) > 6 and value[:6].lower() == "strref" and cls._is_unsigned_decimal(value[6:]):
            return str(memory.memory_str.get(int(value[6:]), 0))
        return value

    @classmethod
    def _resolve_tokens(cls, value: str, memory: PatcherMemory) -> str:
        return cls._resolve_2da_memory(cls._resolve_strref(value, memory), memory)

    @staticmethod
    def _to_internal(value: str) -> str:
        return "" if value == "****" else value

    @staticmethod
    def _to_memory(value: str) -> str:
        return "****" if value == "" else value

    @classmethod
    def _set_cell(cls, twoda: TwoDA, row_index: int, column: str, value: str) -> bool:
        header = cls._find_header(twoda, column)
        if header is None or not 0 <= row_index < twoda.get_height():
            return False
        twoda.get_row(row_index).set_string(header, cls._to_internal(value))
        return True

    @classmethod
    def _capture_2da(
        cls,
        key: str,
        selector: str,
        twoda: TwoDA,
        memory: PatcherMemory,
        *,
        row_index: int | None = None,
        column_header: str | None = None,
    ) -> bool:
        if not key.startswith("2DAMEMORY"):
            return False

        suffix = key[9:]
        if cls._is_unsigned_decimal(suffix):
            token_id = int(suffix)
            if token_id < 1:
                return False
        else:
            token_id = 1

        captured: str | None = None
        if row_index is not None:
            if selector == "RowIndex":
                captured = str(row_index)
            elif selector == "RowLabel":
                if 0 <= row_index < twoda.get_height():
                    captured = twoda.get_label(row_index)
            elif 0 <= row_index < twoda.get_height():
                header = cls._find_header(twoda, selector)
                if header is not None:
                    captured = twoda.get_row(row_index).get_string(header)
        elif column_header is not None:
            if selector == "ColumnLabel":
                captured = column_header
            elif selector:
                row_selector = selector[1:]
                selected_row: int | None = None
                if selector[0].lower() == "i" and cls._is_unsigned_decimal(row_selector):
                    index = int(row_selector)
                    if 0 <= index < twoda.get_height():
                        selected_row = index
                elif selector[0].lower() == "l" and row_selector:
                    selected_row = cls._find_row_index(twoda, row_selector)

                if selected_row is not None:
                    captured = twoda.get_row(selected_row).get_string(column_header)

        if captured is not None:
            memory.memory_2da[token_id] = cls._to_memory(captured)
        return True

    @classmethod
    def _capture_tlk(
        cls,
        key: str,
        selector: str,
        twoda: TwoDA,
        memory: PatcherMemory,
        row_index: int,
    ) -> bool:
        lower_key = key.lower()
        if not lower_key.startswith("strref") or not cls._is_unsigned_decimal(key[6:]):
            return False

        captured: str | None = None
        if selector == "RowIndex":
            captured = str(row_index)
        elif selector == "RowLabel" and 0 <= row_index < twoda.get_height():
            captured = twoda.get_label(row_index)
        elif 0 <= row_index < twoda.get_height():
            header = cls._find_header(twoda, selector)
            if header is not None:
                captured = twoda.get_row(row_index).get_string(header)

        if captured is not None:
            try:
                memory.memory_str[int(key[6:])] = int(captured)
            except ValueError:
                pass
        return True

    @classmethod
    def _exclusive_match(
        cls,
        twoda: TwoDA,
        entries: list[tuple[str, str]],
        exclusive_column: str,
    ) -> tuple[bool, int | None]:
        header = cls._find_header(twoda, exclusive_column)
        if header is None:
            return True, None

        exclusive_value = next(
            (value for key, value in entries if key.lower() == exclusive_column.lower()),
            "",
        )
        if exclusive_value == "":
            return False, None

        for row_index in range(twoda.get_height()):
            if twoda.get_row(row_index).get_string(header) == exclusive_value:
                return False, row_index
        return True, None

    @classmethod
    def _modify_row_fallback(
        cls,
        twoda: TwoDA,
        memory: PatcherMemory,
        entries: list[tuple[str, str]],
        row_index: int | None,
    ):
        if row_index is None or not 0 <= row_index < twoda.get_height():
            return

        for key, raw_value in entries:
            lower_key = key.lower()
            lower_value = raw_value.lower()
            if lower_key in {"rowindex", "rowlabel", "newrowlabel", "exclusivecolumn"}:
                continue
            if lower_value.startswith("high()") or lower_value.startswith("inc("):
                continue
            if cls._capture_2da(key, raw_value, twoda, memory, row_index=row_index):
                continue
            if cls._capture_tlk(key, raw_value, twoda, memory, row_index):
                continue
            if not key:
                continue

            value = raw_value or "****"
            value = cls._resolve_tokens(value, memory)
            cls._set_cell(twoda, row_index, key, value)

    def _unpack(
        self,
        cells: dict[str, RowValue],
        memory: PatcherMemory,
        twoda: TwoDA,
        row: TwoDARow,
    ) -> dict[str, str]:
        return {column: value.value(memory, twoda, row) for column, value in cells.items()}

    @abstractmethod
    def apply(
        self,
        twoda: TwoDA,
        memory: PatcherMemory,
    ): ...


class ChangeRow2DA(Modify2DA):
    """Changes an existing row using the modifier section's original entry order."""

    def __init__(
        self,
        identifier: str,
        target: Target | None = None,
        cells: dict[str, RowValue] | None = None,
        store_2da: dict[int, RowValue] | None = None,
        store_tlk: dict[int, RowValue] | None = None,
        entries: list[tuple[str, str]] | None = None,
    ):
        self.identifier: str = identifier
        self.target: Target | None = target
        self.cells: CaseInsensitiveDict[RowValue] = CaseInsensitiveDict({} if cells is None else cells)
        self.store_2da: dict[int, RowValue] = {} if store_2da is None else store_2da
        self.store_tlk: dict[int, RowValue] = {} if store_tlk is None else store_tlk
        self._row: TwoDARow | None = None

        if entries is None:
            generated: list[tuple[str, str]] = []
            if target is not None:
                generated.append(self._target_to_entry(target))
            generated.extend((column, self._row_value_to_raw(value)) for column, value in self.cells.items())
            generated.extend((f"2DAMEMORY{token_id}", self._row_value_to_raw(value)) for token_id, value in self.store_2da.items())
            generated.extend((f"StrRef{token_id}", self._row_value_to_raw(value)) for token_id, value in self.store_tlk.items())
            entries = generated
        self.entries = self._normalize_entries(entries)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(identifier={self.identifier!r}, "
            f"target={self.target!r}, cells={self.cells!r}, "
            f"store_2da={self.store_2da!r}, store_tlk={self.store_tlk!r}, "
            f"entries={self.entries!r})"
        )

    def apply(self, twoda: TwoDA, memory: PatcherMemory):
        row_index: int | None = None

        for key, raw_value in self.entries:
            lower_key = key.lower()

            if lower_key == "rowindex" and row_index is None and raw_value != "":
                resolved = self._resolve_2da_memory(raw_value, memory)
                try:
                    candidate = int(resolved)
                except ValueError:
                    candidate = -1
                row_index = candidate if 0 <= candidate < twoda.get_height() else None
                continue

            if lower_key == "rowlabel" and row_index is None and raw_value != "":
                resolved = self._resolve_2da_memory(raw_value, memory)
                row_index = self._find_row_index(twoda, resolved)
                continue

            if row_index is None and key == "LabelIndex" and raw_value != "":
                label_header = next((header for header in twoda.get_headers() if header == "label"), None)
                if label_header is None:
                    return
                matching = [
                    index
                    for index in range(twoda.get_height())
                    if twoda.get_row(index).get_string(label_header) == raw_value
                ]
                if matching:
                    row_index = matching[-1]
                    continue
                return

            if row_index is None:
                if lower_key not in {"rowindex", "rowlabel"}:
                    return
                continue

            if self._capture_2da(key, raw_value, twoda, memory, row_index=row_index):
                continue
            if self._capture_tlk(key, raw_value, twoda, memory, row_index):
                continue
            if not key:
                continue

            header = self._find_header(twoda, key)
            if header is None:
                continue

            value = raw_value or "****"
            value = self._resolve_strref(value, memory)
            if value.lower().startswith("high()"):
                value = str(twoda.column_max(header))
            value = self._resolve_2da_memory(value, memory)
            self._set_cell(twoda, row_index, header, value)

        if row_index is not None:
            self._row = twoda.get_row(row_index)


class AddRow2DA(Modify2DA):
    """Adds a row lazily while processing the modifier section in order."""

    def __init__(
        self,
        identifier: str,
        exclusive_column: str | None = None,
        row_label: RowValue | str | None = None,
        cells: dict[str, RowValue] | None = None,
        store_2da: dict[int, RowValue] | None = None,
        store_tlk: dict[int, RowValue] | None = None,
        entries: list[tuple[str, str]] | None = None,
    ):
        self.identifier: str = identifier
        self.exclusive_column: str | None = exclusive_column or None
        self.row_label: RowValue | str | None = row_label
        self.cells: CaseInsensitiveDict[RowValue] = CaseInsensitiveDict({} if cells is None else cells)
        self.store_2da: dict[int, RowValue] = {} if store_2da is None else store_2da
        self.store_tlk: dict[int, RowValue] = {} if store_tlk is None else store_tlk
        self._row: TwoDARow | None = None

        if entries is None:
            generated: list[tuple[str, str]] = []
            if self.exclusive_column is not None:
                generated.append(("ExclusiveColumn", self.exclusive_column))
            if row_label is not None:
                generated.append(("RowLabel", self._row_value_to_raw(row_label)))
            generated.extend((column, self._row_value_to_raw(value)) for column, value in self.cells.items())
            generated.extend((f"2DAMEMORY{token_id}", self._row_value_to_raw(value)) for token_id, value in self.store_2da.items())
            generated.extend((f"StrRef{token_id}", self._row_value_to_raw(value)) for token_id, value in self.store_tlk.items())
            entries = generated
        self.entries = self._normalize_entries(entries)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(identifier={self.identifier!r}, "
            f"exclusive_column={self.exclusive_column!r}, row_label={self.row_label!r}, "
            f"cells={self.cells!r}, store_2da={self.store_2da!r}, "
            f"store_tlk={self.store_tlk!r}, entries={self.entries!r})"
        )

    def apply(self, twoda: TwoDA, memory: PatcherMemory):
        exclusive_column = next(
            (value for key, value in self.entries if key.lower() == "exclusivecolumn" and value != ""),
            None,
        )
        if exclusive_column is not None:
            should_add, existing_row = self._exclusive_match(twoda, self.entries, exclusive_column)
            if not should_add:
                self._modify_row_fallback(twoda, memory, self.entries, existing_row)
                if existing_row is not None:
                    self._row = twoda.get_row(existing_row)
                return

        row_index: int | None = None
        for key, raw_value in self.entries:
            lower_key = key.lower()
            if lower_key == "exclusivecolumn":
                continue

            if lower_key == "rowlabel":
                value = raw_value
                if value.lower().startswith("high()"):
                    value = str(twoda.label_max())
                value = self._resolve_2da_memory(value, memory)
                if value != "":
                    if row_index is None:
                        row_index = twoda.add_row(str(twoda.get_height()), {})
                    twoda.set_label(row_index, value)
                continue

            if row_index is not None and self._capture_2da(key, raw_value, twoda, memory, row_index=row_index):
                continue
            if row_index is not None and self._capture_tlk(key, raw_value, twoda, memory, row_index):
                continue

            if row_index is None:
                row_index = twoda.add_row(str(twoda.get_height()), {})

            header = self._find_header(twoda, key)
            if header is None:
                continue

            value = raw_value or "****"
            value = self._resolve_strref(value, memory)
            if value.lower().startswith("high()"):
                value = str(twoda.column_max(header))
            value = self._resolve_2da_memory(value, memory)
            self._set_cell(twoda, row_index, header, value)

        if row_index is not None:
            self._row = twoda.get_row(row_index)


class CopyRow2DA(Modify2DA):
    """Copies a row lazily while processing the modifier section in order."""

    def __init__(
        self,
        identifier: str,
        target: Target | None = None,
        exclusive_column: str | None = None,
        row_label: RowValue | str | None = None,
        cells: dict[str, RowValue] | None = None,
        store_2da: dict[int, RowValue] | None = None,
        store_tlk: dict[int, RowValue] | None = None,
        entries: list[tuple[str, str]] | None = None,
    ):
        self.identifier: str = identifier
        self.target: Target | None = target
        self.exclusive_column: str | None = exclusive_column or None
        self.row_label: RowValue | str | None = row_label
        self.cells: CaseInsensitiveDict[RowValue] = CaseInsensitiveDict({} if cells is None else cells)
        self.store_2da: dict[int, RowValue] = {} if store_2da is None else store_2da
        self.store_tlk: dict[int, RowValue] = {} if store_tlk is None else store_tlk
        self._row: TwoDARow | None = None

        if entries is None:
            generated: list[tuple[str, str]] = []
            if target is not None:
                generated.append(self._target_to_entry(target))
            if self.exclusive_column is not None:
                generated.append(("ExclusiveColumn", self.exclusive_column))
            if row_label is not None:
                generated.append(("NewRowLabel", self._row_value_to_raw(row_label)))
            generated.extend((column, self._row_value_to_raw(value)) for column, value in self.cells.items())
            generated.extend((f"2DAMEMORY{token_id}", self._row_value_to_raw(value)) for token_id, value in self.store_2da.items())
            generated.extend((f"StrRef{token_id}", self._row_value_to_raw(value)) for token_id, value in self.store_tlk.items())
            entries = generated
        self.entries = self._normalize_entries(entries)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(identifier={self.identifier!r}, "
            f"target={self.target!r}, exclusive_column={self.exclusive_column!r}, "
            f"row_label={self.row_label!r}, cells={self.cells!r}, "
            f"store_2da={self.store_2da!r}, store_tlk={self.store_tlk!r}, "
            f"entries={self.entries!r})"
        )

    def apply(self, twoda: TwoDA, memory: PatcherMemory):
        exclusive_column = next(
            (value for key, value in self.entries if key.lower() == "exclusivecolumn" and value != ""),
            None,
        )
        if exclusive_column is not None:
            should_copy, existing_row = self._exclusive_match(twoda, self.entries, exclusive_column)
            if not should_copy:
                self._modify_row_fallback(twoda, memory, self.entries, existing_row)
                if existing_row is not None:
                    self._row = twoda.get_row(existing_row)
                return

        row_index: int | None = None
        cloned = False
        new_row_label = ""

        for key, raw_value in self.entries:
            lower_key = key.lower()

            if lower_key == "rowindex" and row_index is None and self._is_unsigned_decimal(raw_value):
                resolved = self._resolve_2da_memory(raw_value, memory)
                candidate = int(resolved)
                row_index = candidate if 0 <= candidate < twoda.get_height() else None
                continue

            if lower_key == "rowlabel" and row_index is None and raw_value != "":
                resolved = self._resolve_2da_memory(raw_value, memory)
                row_index = self._find_row_index(twoda, resolved)
                continue

            if lower_key == "exclusivecolumn":
                continue

            if row_index is not None and self._capture_2da(key, raw_value, twoda, memory, row_index=row_index):
                continue
            if row_index is not None and self._capture_tlk(key, raw_value, twoda, memory, row_index):
                continue

            if lower_key == "newrowlabel" and raw_value != "":
                value = raw_value
                if value.lower().startswith("high()"):
                    value = str(twoda.label_max())
                new_row_label = self._resolve_2da_memory(value, memory)
                continue

            if row_index is None or not key:
                continue

            value = raw_value or "****"
            value = self._resolve_strref(value, memory)

            if not cloned:
                source_row = twoda.get_row(row_index)
                label = new_row_label if new_row_label != "" else str(twoda.get_height())
                row_index = twoda.copy_row(source_row, label, {})
                cloned = True

            header = self._find_header(twoda, key)
            if header is None:
                continue

            current_value = twoda.get_row(row_index).get_string(header)
            lower_value = value.lower()
            if lower_value.startswith("inc(") and value.endswith(")"):
                closing = value.find(")")
                increment = value[4:closing]
                if self._is_unsigned_decimal(current_value) and self._is_unsigned_decimal(increment):
                    value = str(int(current_value) + int(increment))
                else:
                    value = current_value
            elif lower_value.startswith("high()"):
                value = str(twoda.column_max(header))

            value = self._resolve_2da_memory(value, memory)
            self._set_cell(twoda, row_index, header, value)

        if cloned and row_index is not None:
            self._row = twoda.get_row(row_index)


class AddColumn2DA(Modify2DA):
    """Adds a column and applies its modifier entries in their original order."""

    def __init__(
        self,
        identifier: str,
        header: str = "",
        default: str = "",
        index_insert: dict[int, RowValue] | None = None,
        label_insert: dict[str, RowValue] | None = None,
        store_2da: dict[int, str] | None = None,
        entries: list[tuple[str, str]] | None = None,
    ):
        self.identifier: str = identifier
        self.header: str = header
        self.default: str = default
        self.index_insert: dict[int, RowValue] = {} if index_insert is None else index_insert
        self.label_insert: dict[str, RowValue] = {} if label_insert is None else label_insert
        self.store_2da: dict[int, str] = {} if store_2da is None else store_2da

        if entries is None:
            generated: list[tuple[str, str]] = [("ColumnLabel", header), ("DefaultValue", "****" if default == "" else default)]
            generated.extend((f"I{row_index}", self._row_value_to_raw(value)) for row_index, value in self.index_insert.items())
            generated.extend((f"L{row_label}", self._row_value_to_raw(value)) for row_label, value in self.label_insert.items())
            generated.extend((f"2DAMEMORY{token_id}", selector) for token_id, selector in self.store_2da.items())
            entries = generated
        self.entries = self._normalize_entries(entries)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(identifier={self.identifier!r}, "
            f"header={self.header!r}, default={self.default!r}, "
            f"index_insert={self.index_insert}, label_insert={self.label_insert}, "
            f"store_2da={self.store_2da}, entries={self.entries!r})"
        )

    def apply(self, twoda: TwoDA, memory: PatcherMemory):
        added = False
        header = ""
        default = ""

        for key, raw_value in self.entries:
            lower_key = key.lower()

            if lower_key == "columnlabel" and not added:
                if raw_value == "":
                    continue
                resolved_header = self._resolve_2da_memory(raw_value, memory)
                if resolved_header in twoda.get_headers():
                    return
                twoda.add_column(resolved_header)
                header = resolved_header
                added = True
                continue

            if added and self._capture_2da(key, raw_value, twoda, memory, column_header=header):
                continue

            if lower_key == "defaultvalue" and added:
                if raw_value == "":
                    continue
                value = self._resolve_tokens(raw_value, memory)
                value = self._to_internal(value)
                old_default = default
                default = value
                for row_index in range(twoda.get_height()):
                    row = twoda.get_row(row_index)
                    if row.get_string(header) == old_default:
                        row.set_string(header, default)
                continue

            if not added or not key:
                continue

            value = default if raw_value == "" else self._resolve_strref(raw_value, memory)
            row_selector = key[1:]
            if key[0].lower() == "i" and self._is_unsigned_decimal(row_selector):
                row_index = int(row_selector)
                if 0 <= row_index < twoda.get_height():
                    self._set_cell(twoda, row_index, header, value)
            elif key[0].lower() == "l" and row_selector:
                row_index = self._find_row_index(twoda, row_selector)
                if row_index is not None:
                    value = self._resolve_2da_memory(value, memory)
                    self._set_cell(twoda, row_index, header, value)


# endregion


class Modifications2DA(PatcherModifications):
    def __init__(self, filename: str):
        super().__init__(filename)
        self.modifiers: list[Modify2DA] = []

    def patch_resource(
        self,
        source_2da: SOURCE_TYPES,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ) -> bytes | Literal[True]:
        twoda: TwoDA = read_2da(source_2da)
        self.apply(twoda, memory, logger, game)
        return bytes_2da(twoda)

    def apply(
        self,
        twoda: TwoDA,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ):
        for row in self.modifiers:
            try:
                row.apply(twoda, memory)
            except Exception as e:  # noqa: PERF203, BLE001
                msg = f"{universal_simplify_exception(e)} when patching the file '{self.saveas}'"
                RobustRootLogger().critical(str(e), exc_info=e)
                if isinstance(e, WarningError):
                    logger.add_warning(msg)
                    RobustRootLogger().debug(msg, exc_info=True)
                else:
                    logger.add_error(msg)
                    break
