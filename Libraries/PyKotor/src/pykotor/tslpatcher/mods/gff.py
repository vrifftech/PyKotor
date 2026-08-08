from __future__ import annotations

import base64
import binascii
import re

from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pykotor.common.geometry import Vector3, Vector4
from pykotor.common.language import LocalizedString
from pykotor.common.misc import ResRef
from pykotor.common.stream import BinaryReader
from pykotor.resource.formats.gff import GFFFieldType, GFFList, GFFStruct, bytes_gff
from pykotor.resource.formats.gff.gff_data import _GFFField
from pykotor.resource.formats.gff.io_gff import GFFBinaryReader
from pykotor.tslpatcher.mods.template import PatcherModifications
from utility.system.path import PureWindowsPath

if TYPE_CHECKING:
    import os

    from collections.abc import Callable

    from typing_extensions import Literal

    from pykotor.common.misc import Game
    from pykotor.resource.formats.gff import GFF
    from pykotor.resource.type import SOURCE_TYPES
    from pykotor.tslpatcher.logger import PatchLogger
    from pykotor.tslpatcher.memory import PatcherMemory


_INVALID = object()
_ASCII_DIGITS = re.compile(r"^[0-9]+$")
_SIGNED_INTEGER = re.compile(r"^-?[0-9]+$")
_FLOAT = re.compile(r"^-?[0-9]+(?:[.,][0-9]+)?$")

_INTEGER_RANGES: dict[GFFFieldType, tuple[int, int]] = {
    GFFFieldType.UInt8: (0, 0xFF),
    GFFFieldType.Int8: (-0x80, 0x7F),
    GFFFieldType.UInt16: (0, 0xFFFF),
    GFFFieldType.Int16: (-0x8000, 0x7FFF),
    GFFFieldType.UInt32: (0, 0xFFFFFFFF),
    GFFFieldType.Int32: (-0x80000000, 0x7FFFFFFF),
    GFFFieldType.UInt64: (0, 0xFFFFFFFFFFFFFFFF),
    GFFFieldType.Int64: (-0x8000000000000000, 0x7FFFFFFFFFFFFFFF),
}


def _clone_locstring(value: LocalizedString) -> LocalizedString:
    return LocalizedString(value.stringref, dict(value._substrings))


def _parse_int(value: Any, field_type: GFFFieldType, *, allow_negative: bool) -> int | object:
    if isinstance(value, bool):
        return _INVALID
    if isinstance(value, int):
        parsed = value
    else:
        text = str(value).strip()
        pattern = _SIGNED_INTEGER if allow_negative else _ASCII_DIGITS
        if pattern.fullmatch(text) is None:
            return _INVALID
        try:
            parsed = int(text, 10)
        except ValueError:
            return _INVALID

    minimum, maximum = _INTEGER_RANGES[field_type]
    return parsed if minimum <= parsed <= maximum else _INVALID


def _parse_float(value: Any) -> float | object:
    if isinstance(value, bool):
        return _INVALID
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if _FLOAT.fullmatch(text) is None:
        return _INVALID
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return _INVALID


def _parse_binary(value: Any) -> bytes | object:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()

    text = str(value).strip()
    if not text:
        return b""
    if set(text) <= {"0", "1"} and len(text) % 8 == 0:
        try:
            return bytes(int(text[offset : offset + 8], 2) for offset in range(0, len(text), 8))
        except ValueError:
            return _INVALID
    if text.lower().startswith("0x"):
        hex_string = text[2:]
        if len(hex_string) % 2:
            hex_string = f"0{hex_string}"
        try:
            return bytes.fromhex(hex_string)
        except ValueError:
            return _INVALID
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError, TypeError):
        return _INVALID


def _parse_vector(
    value: Any,
    field_type: GFFFieldType,
    existing: Vector3 | Vector4 | None = None,
) -> Vector3 | Vector4 | object:
    count = 3 if field_type is GFFFieldType.Vector3 else 4
    vector_cls = Vector3 if count == 3 else Vector4
    if isinstance(value, vector_cls):
        return copy(value)
    if not isinstance(value, str):
        return _INVALID

    components = value.split("|")
    if len(components) != count:
        return _INVALID

    if existing is None:
        parsed = [_parse_float(component) for component in components]
        if any(component is _INVALID for component in parsed):
            return _INVALID
        return vector_cls(*parsed)

    result = copy(existing)
    component_names = ("x", "y", "z") if count == 3 else ("x", "y", "z", "w")
    for component_name, component in zip(component_names, components):
        parsed_component = _parse_float(component)
        if parsed_component is not _INVALID:
            setattr(result, component_name, parsed_component)
    return result


def _parse_char(value: Any, *, empty_value: int | object) -> int | object:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if -0x80 <= value <= 0x7F else _INVALID
    text = str(value)
    if not text:
        return empty_value
    codepoint = ord(text[0])
    if codepoint > 0x7F:
        codepoint -= 0x100
    return codepoint if -0x80 <= codepoint <= 0x7F else _INVALID


def _coerce_scalar(
    value: Any,
    field_type: GFFFieldType,
    *,
    existing: Any = None,
    new_field: bool,
) -> Any:
    if field_type is GFFFieldType.Int8:
        return _parse_char(value, empty_value=_INVALID if new_field else 0)
    if field_type in _INTEGER_RANGES:
        signed_type = field_type in {GFFFieldType.Int16, GFFFieldType.Int32, GFFFieldType.Int64}
        return _parse_int(value, field_type, allow_negative=new_field and signed_type)
    if field_type in {GFFFieldType.Single, GFFFieldType.Double}:
        return _parse_float(value)
    if field_type is GFFFieldType.String:
        return str(value).replace("<#LF#>", "\n").replace("<#CR#>", "\r")
    if field_type is GFFFieldType.ResRef:
        return value if isinstance(value, ResRef) else ResRef(str(value)) if str(value) else ResRef.from_blank()
    if field_type in {GFFFieldType.Vector3, GFFFieldType.Vector4}:
        return _parse_vector(value, field_type, existing)
    if field_type is GFFFieldType.Binary:
        return _parse_binary(value)
    return _INVALID


def set_locstring(
    struct: GFFStruct,
    label: str,
    value: LocalizedStringDelta,
    memory: PatcherMemory,
):
    original = struct.get_locstring(label) if struct.exists(label) and struct._fields[label].field_type() is GFFFieldType.LocalizedString else LocalizedString(-1)
    patched = _clone_locstring(original)
    value.apply(patched, memory, invalid_stringref=-1)
    struct.set_locstring(label, patched)


FIELD_TYPE_TO_GETTER: dict[GFFFieldType, Callable[[GFFStruct, str], Any]] = {
    GFFFieldType.Int8: GFFStruct.get_int8,
    GFFFieldType.UInt8: GFFStruct.get_uint8,
    GFFFieldType.Int16: GFFStruct.get_int16,
    GFFFieldType.UInt16: GFFStruct.get_uint16,
    GFFFieldType.Int32: GFFStruct.get_int32,
    GFFFieldType.UInt32: GFFStruct.get_uint32,
    GFFFieldType.Int64: GFFStruct.get_int64,
    GFFFieldType.UInt64: GFFStruct.get_uint64,
    GFFFieldType.Single: GFFStruct.get_single,
    GFFFieldType.Double: GFFStruct.get_double,
    GFFFieldType.String: GFFStruct.get_string,
    GFFFieldType.ResRef: GFFStruct.get_resref,
    GFFFieldType.LocalizedString: GFFStruct.get_locstring,
    GFFFieldType.Binary: GFFStruct.get_binary,
    GFFFieldType.Vector3: GFFStruct.get_vector3,
    GFFFieldType.Vector4: GFFStruct.get_vector4,
    GFFFieldType.List: GFFStruct.get_list,
    GFFFieldType.Struct: GFFStruct.get_struct,
}


FIELD_TYPE_TO_SETTER: dict[GFFFieldType, Callable[[GFFStruct, str, Any, PatcherMemory], None]] = {
    GFFFieldType.Int8: lambda s, lbl, v, _m: s.set_int8(lbl, v),
    GFFFieldType.UInt8: lambda s, lbl, v, _m: s.set_uint8(lbl, v),
    GFFFieldType.Int16: lambda s, lbl, v, _m: s.set_int16(lbl, v),
    GFFFieldType.UInt16: lambda s, lbl, v, _m: s.set_uint16(lbl, v),
    GFFFieldType.Int32: lambda s, lbl, v, _m: s.set_int32(lbl, v),
    GFFFieldType.UInt32: lambda s, lbl, v, _m: s.set_uint32(lbl, v),
    GFFFieldType.Int64: lambda s, lbl, v, _m: s.set_int64(lbl, v),
    GFFFieldType.UInt64: lambda s, lbl, v, _m: s.set_uint64(lbl, v),
    GFFFieldType.Single: lambda s, lbl, v, _m: s.set_single(lbl, v),
    GFFFieldType.Double: lambda s, lbl, v, _m: s.set_double(lbl, v),
    GFFFieldType.String: lambda s, lbl, v, _m: s.set_string(lbl, v),
    GFFFieldType.ResRef: lambda s, lbl, v, _m: s.set_resref(lbl, v),
    GFFFieldType.LocalizedString: set_locstring,
    GFFFieldType.Binary: lambda s, lbl, v, _m: s.set_binary(lbl, v),
    GFFFieldType.Vector3: lambda s, lbl, v, _m: s.set_vector3(lbl, v),
    GFFFieldType.Vector4: lambda s, lbl, v, _m: s.set_vector4(lbl, v),
    GFFFieldType.List: lambda s, lbl, v, _m: s.set_list(lbl, v),
    GFFFieldType.Struct: lambda s, lbl, v, _m: s.set_struct(lbl, v),
}


class FieldValue(ABC):
    @abstractmethod
    def resolve(self, memory: PatcherMemory) -> Any: ...

    def value(self, memory: PatcherMemory, field_type: GFFFieldType) -> Any:
        resolved = self.resolve(memory)
        if isinstance(resolved, PureWindowsPath):
            return resolved
        if field_type is GFFFieldType.ResRef:
            return resolved if isinstance(resolved, ResRef) else ResRef(str(resolved)) if str(resolved) else ResRef.from_blank()
        if field_type is GFFFieldType.String:
            return str(resolved)
        if field_type in _INTEGER_RANGES:
            parsed = _parse_int(resolved, field_type, allow_negative=True)
            if parsed is _INVALID:
                raise ValueError(f"Invalid {field_type.name} value: {resolved!r}")
            return parsed
        if field_type in {GFFFieldType.Single, GFFFieldType.Double}:
            parsed = _parse_float(resolved)
            if parsed is _INVALID:
                raise ValueError(f"Invalid {field_type.name} value: {resolved!r}")
            return parsed
        return resolved


class FieldValueConstant(FieldValue):
    def __init__(self, value: Any):
        self.stored: Any = value

    def resolve(self, memory: PatcherMemory) -> Any:
        return self.stored


class FieldValueRaw(FieldValueConstant):
    """An INI value retained verbatim until the destination field type is known."""

    def __init__(self, raw_value: str, preview: Any = _INVALID):
        self.raw_value = raw_value
        super().__init__(raw_value if preview is _INVALID else preview)

    def resolve(self, memory: PatcherMemory) -> Any:
        raw_value = self.raw_value
        lower_value = raw_value.lower()

        if lower_value.startswith("strref") and _ASCII_DIGITS.fullmatch(raw_value[6:]):
            token_id = int(raw_value[6:])
            return memory.memory_str.get(token_id, 0)

        if raw_value.startswith("2DAMEMORY"):
            if not memory.memory_2da:
                return raw_value
            suffix = raw_value[9:]
            token_id = int(suffix) if _ASCII_DIGITS.fullmatch(suffix) else 1
            if token_id not in memory.memory_2da:
                token_id = 1
            return memory.memory_2da.get(token_id, raw_value)

        return raw_value

    def value(self, memory: PatcherMemory | None, field_type: GFFFieldType) -> Any:
        if memory is None:
            return deepcopy(self.stored)
        return super().value(memory, field_type)


class FieldValueListIndex(FieldValueConstant):
    pass


class FieldValue2DAMemory(FieldValue):
    def __init__(self, token_id: int):
        self.token_id: int = token_id

    def resolve(self, memory: PatcherMemory) -> Any:
        token = f"2DAMEMORY{self.token_id}"
        if not memory.memory_2da:
            return token
        token_id = self.token_id if self.token_id in memory.memory_2da else 1
        return memory.memory_2da.get(token_id, token)


class FieldValueTLKMemory(FieldValue):
    def __init__(self, token_id: int):
        self.token_id: int = token_id

    def resolve(self, memory: PatcherMemory) -> Any:
        return memory.memory_str.get(self.token_id, 0)


class LocalizedStringDelta(LocalizedString):
    def __init__(self, stringref: FieldValue | None = None):
        super().__init__(-1)
        self.stringref: FieldValue | None = stringref  # type: ignore[assignment]
        self._deferred_substrings: dict[int, FieldValue] = {}

    def __str__(self):
        return f"LocalizedStringDelta(stringref={self.stringref!r})"

    def set_field_value(self, substring_id: int, value: FieldValue):
        self._deferred_substrings[substring_id] = value

    def apply(
        self,
        locstring: LocalizedString,
        memory: PatcherMemory,
        *,
        invalid_stringref: int | None = None,
    ) -> bool:
        changed = False
        if self.stringref is not None:
            raw_stringref = self.stringref.resolve(memory)
            parsed_stringref = _parse_int(raw_stringref, GFFFieldType.Int32, allow_negative=True)
            if parsed_stringref is _INVALID:
                parsed_stringref = invalid_stringref
            if parsed_stringref is not None and locstring.stringref != parsed_stringref:
                locstring.stringref = parsed_stringref
                changed = True

        for language, gender, text in self:
            if locstring.get(language, gender) != text:
                locstring.set_data(language, gender, text)
                changed = True

        for substring_id, field_value in self._deferred_substrings.items():
            raw_text = field_value.resolve(memory)
            text = str(raw_text).replace("<#LF#>", "\n").replace("<#CR#>", "\r")
            language, gender = self.substring_pair(substring_id)
            if locstring.get(language, gender) != text:
                locstring.set_data(language, gender, text)
                changed = True
        return changed


@dataclass(frozen=True)
class GFFModifierContext:
    path: PureWindowsPath = PureWindowsPath("")
    list_index: int | None = None


class ModifyGFF(ABC):
    @abstractmethod
    def apply(
        self,
        root_container: GFFStruct | GFFList,
        memory: PatcherMemory,
        logger: PatchLogger,
        context: GFFModifierContext | None = None,
    ) -> bool: ...

    @staticmethod
    def _resolve_path(path: PureWindowsPath, relative: bool, context: GFFModifierContext | None) -> PureWindowsPath:
        if relative and context is not None:
            return context.path / path
        return path

    def _navigate_containers(
        self,
        root_container: GFFStruct,
        path: PureWindowsPath | os.PathLike | str,
    ) -> GFFList | GFFStruct | None:
        path = PureWindowsPath.pathify(path)
        if not path.name:
            return root_container
        container: GFFStruct | GFFList | None = root_container
        for step in path.parts:
            if isinstance(container, GFFStruct):
                container = container.acquire(step, None, (GFFStruct, GFFList))
            elif isinstance(container, GFFList):
                if _ASCII_DIGITS.fullmatch(step) is None:
                    return None
                index = int(step)
                container = container.at(index) if 0 <= index < len(container) else None
            if container is None:
                return None
        return container

    def _navigate_to_field(
        self,
        root_container: GFFStruct,
        path: PureWindowsPath | os.PathLike | str,
    ) -> _GFFField | None:
        path = PureWindowsPath.pathify(path)
        container = self._navigate_containers(root_container, path.parent)
        return container._fields.get(path.name) if isinstance(container, GFFStruct) else None

    def _resolve_field_pointer(self, root_struct: GFFStruct, value: Any) -> Any:
        if not isinstance(value, PureWindowsPath):
            return value
        field = self._navigate_to_field(root_struct, value)
        return deepcopy(field.value()) if field is not None else _INVALID

    @staticmethod
    def _apply_modifiers(
        modifiers: list[ModifyGFF],
        root_struct: GFFStruct,
        memory: PatcherMemory,
        logger: PatchLogger,
        context: GFFModifierContext,
    ) -> bool:
        changed = False
        for modifier in modifiers:
            try:
                changed = modifier.apply(root_struct, memory, logger, context) or changed
            except Exception as exc:  # noqa: BLE001 - one malformed INI operation must not abort later operations.
                logger.add_error(f"Unable to apply GFF modifier [{getattr(modifier, 'identifier', '')}]: {exc}")
        return changed


class AddStructToListGFF(ModifyGFF):
    def __init__(
        self,
        identifier: str,
        value: FieldValue,
        path: PureWindowsPath | os.PathLike | str,
        index_to_token: int | None = None,
        modifiers: list[ModifyGFF] | None = None,
        *,
        relative_path: bool = False,
    ):
        self.identifier = identifier
        self.value = value
        self.path = PureWindowsPath.pathify(path)
        self.relative_path = relative_path
        self._index_to_token = index_to_token
        self.modifiers = [] if modifiers is None else modifiers

    @property
    def index_to_token(self) -> int | None:
        if self._index_to_token is not None:
            return self._index_to_token
        return next(
            (
                modifier.dest_token_id
                for modifier in self.modifiers
                if isinstance(modifier, Memory2DAModifierGFF) and modifier.store_list_index
            ),
            None,
        )

    def apply(
        self,
        root_struct: GFFStruct,
        memory: PatcherMemory,
        logger: PatchLogger,
        context: GFFModifierContext | None = None,
    ) -> bool:
        path = self._resolve_path(self.path, self.relative_path, context)
        list_container = self._navigate_containers(root_struct, path)
        if not isinstance(list_container, GFFList):
            reason = "does not exist" if list_container is None else f"is a {type(list_container).__name__}, not a GFFList"
            logger.add_error(f"Unable to add struct to list '{path}' in [{self.identifier}]: path {reason}.")
            return False

        raw_value = self._resolve_field_pointer(root_struct, self.value.resolve(memory))
        if isinstance(raw_value, GFFStruct):
            new_struct = deepcopy(raw_value)
        else:
            raw_type_id = str(raw_value).strip()
            if raw_type_id.lower() == "listindex":
                type_id = len(list_container)
            elif _ASCII_DIGITS.fullmatch(raw_type_id):
                type_id = int(raw_type_id)
            else:
                type_id = 0
            new_struct = GFFStruct(type_id)

        list_container._structs.append(new_struct)
        list_index = len(list_container) - 1
        if self._index_to_token is not None:
            memory.memory_2da[self._index_to_token] = str(list_index)

        child_context = GFFModifierContext(path / str(list_index), list_index)
        self._apply_modifiers(self.modifiers, root_struct, memory, logger, child_context)
        return True


class AddFieldGFF(ModifyGFF):
    def __init__(
        self,
        identifier: str,
        label: str,
        field_type: GFFFieldType,
        value: FieldValue,
        path: PureWindowsPath | os.PathLike | str,
        modifiers: list[ModifyGFF] | None = None,
        *,
        relative_path: bool = False,
    ):
        self.identifier = identifier
        self.label = label
        self.field_type = field_type
        self.value = value
        self.path = PureWindowsPath.pathify(path)
        self.relative_path = relative_path
        self.modifiers = [] if modifiers is None else modifiers

    def apply(
        self,
        root_struct: GFFStruct,
        memory: PatcherMemory,
        logger: PatchLogger,
        context: GFFModifierContext | None = None,
    ) -> bool:
        path = self._resolve_path(self.path, self.relative_path, context)
        struct_container = self._navigate_containers(root_struct, path)
        if not isinstance(struct_container, GFFStruct):
            reason = "does not exist" if struct_container is None else f"is a {type(struct_container).__name__}, not a GFFStruct"
            logger.add_error(f"Unable to add GFF field '{self.label}' at '{path}' in [{self.identifier}]: path {reason}.")
            return False

        existing_field = struct_container._fields.get(self.label)
        if existing_field is not None and existing_field.field_type() is not self.field_type:
            logger.add_warning(
                f"Unable to add {self.field_type.name} field '{self.label}' in [{self.identifier}]: "
                f"an existing {existing_field.field_type().name} field uses that label."
            )
            return False

        changed = False
        field_path = path / self.label

        if self.field_type is GFFFieldType.List:
            if existing_field is None:
                raw_value = self._resolve_field_pointer(root_struct, self.value.resolve(memory))
                value = deepcopy(raw_value) if isinstance(raw_value, GFFList) else GFFList()
                struct_container.set_list(self.label, value)
                changed = True

        elif self.field_type is GFFFieldType.Struct:
            raw_value = self._resolve_field_pointer(root_struct, self.value.resolve(memory))
            if existing_field is None:
                if isinstance(raw_value, GFFStruct):
                    value = deepcopy(raw_value)
                else:
                    raw_type_id = str(raw_value).strip()
                    type_id = int(raw_type_id) if _ASCII_DIGITS.fullmatch(raw_type_id) else 0
                    value = GFFStruct(type_id)
                struct_container.set_struct(self.label, value)
                changed = True
            else:
                existing_struct = existing_field.value()
                if isinstance(raw_value, GFFStruct):
                    type_id = raw_value.struct_id
                else:
                    raw_type_id = str(raw_value).strip()
                    type_id = int(raw_type_id) if _ASCII_DIGITS.fullmatch(raw_type_id) else existing_struct.struct_id
                if existing_struct.struct_id != type_id:
                    existing_struct.struct_id = type_id
                    changed = True

        elif self.field_type is GFFFieldType.LocalizedString:
            raw_value = self.value.resolve(memory)
            if not isinstance(raw_value, LocalizedStringDelta):
                logger.add_error(f"Invalid localized-string value in [{self.identifier}].")
                return False
            if existing_field is None:
                value = LocalizedString(-1)
                raw_value.apply(value, memory, invalid_stringref=-1)
                struct_container.set_locstring(self.label, value)
                changed = True
            else:
                original = struct_container.get_locstring(self.label)
                patched = _clone_locstring(original)
                if raw_value.apply(patched, memory, invalid_stringref=-1):
                    struct_container.set_locstring(self.label, patched)
                    changed = True

        else:
            raw_value = self._resolve_field_pointer(root_struct, self.value.resolve(memory))
            if raw_value is _INVALID:
                return False
            existing_value = existing_field.value() if existing_field is not None else None
            parsed_value = _coerce_scalar(
                raw_value,
                self.field_type,
                existing=existing_value,
                new_field=existing_field is None,
            )
            if parsed_value is _INVALID:
                logger.add_warning(f"Invalid {self.field_type.name} value '{raw_value}' in [{self.identifier}]; skipping field.")
                return False
            if existing_field is None or existing_value != parsed_value:
                FIELD_TYPE_TO_SETTER[self.field_type](struct_container, self.label, parsed_value, memory)
                changed = True

        child_context = GFFModifierContext(field_path, context.list_index if context else None)
        return self._apply_modifiers(self.modifiers, root_struct, memory, logger, child_context) or changed


class Memory2DAModifierGFF(ModifyGFF):
    def __init__(
        self,
        identifier: str,
        path: PureWindowsPath | os.PathLike | str,
        dst_token_id: int,
        src_token_id: int | None = None,
        *,
        relative_path: bool = False,
        store_list_index: bool = False,
    ):
        self.identifier = identifier
        self.dest_token_id = dst_token_id
        self.src_token_id = src_token_id
        self.path = PureWindowsPath.pathify(path)
        self.relative_path = relative_path
        self.store_list_index = store_list_index

    def apply(
        self,
        root_struct: GFFStruct,
        memory: PatcherMemory,
        logger: PatchLogger,
        context: GFFModifierContext | None = None,
    ) -> bool:
        if self.store_list_index:
            if context is None or context.list_index is None:
                logger.add_error(f"Cannot assign 2DAMEMORY{self.dest_token_id}=ListIndex outside a list item.")
                return False
            memory.memory_2da[self.dest_token_id] = str(context.list_index)
            return False

        path = self._resolve_path(self.path, self.relative_path, context)
        if self.src_token_id is None:
            memory.memory_2da[self.dest_token_id] = path
            return False

        source = memory.memory_2da.get(self.src_token_id)
        if source is None:
            logger.add_error(f"2DAMEMORY{self.src_token_id} was not defined before use.")
            return False

        destination = memory.memory_2da.get(self.dest_token_id)
        if isinstance(destination, PureWindowsPath):
            field = self._navigate_to_field(root_struct, destination)
            if field is None:
                logger.add_error(f"Stored field path '{destination}' does not point to a GFF field.")
                return False
            source_value = self._resolve_field_pointer(root_struct, source)
            parsed_value = _coerce_scalar(source_value, field.field_type(), existing=field.value(), new_field=False)
            if parsed_value is _INVALID:
                return False
            if field.value() != parsed_value:
                field._value = parsed_value
                return True
            return False

        memory.memory_2da[self.dest_token_id] = source
        return False


class ModifyFieldGFF(ModifyGFF):
    def __init__(
        self,
        path: PureWindowsPath | os.PathLike | str,
        value: FieldValue,
        identifier: str = "",
        *,
        path_token_id: int | None = None,
        relative_path: bool = False,
    ):
        self.path = PureWindowsPath.pathify(path)
        self.value = value
        self.identifier = identifier
        self.path_token_id = path_token_id
        self.relative_path = relative_path

    def apply(
        self,
        root_struct: GFFStruct,
        memory: PatcherMemory,
        logger: PatchLogger,
        context: GFFModifierContext | None = None,
    ) -> bool:
        if self.path_token_id is not None:
            stored_path = memory.memory_2da.get(self.path_token_id)
            if stored_path is None:
                logger.add_error(f"2DAMEMORY{self.path_token_id} does not contain a GFF field path.")
                return False
            path = PureWindowsPath.pathify(stored_path)
        else:
            path = self._resolve_path(self.path, self.relative_path, context)

        label = path.name
        navigated_struct = self._navigate_containers(root_struct, path.parent)
        if not isinstance(navigated_struct, GFFStruct):
            logger.add_error(f"Unable to modify GFF field '{label}': parent path '{path.parent}' was not found.")
            return False

        field = navigated_struct._fields.get(label)
        if field is None:
            logger.add_error(f"Unable to modify missing GFF field '{path}' in [{self.identifier}].")
            return False

        field_type = field.field_type()
        if field_type in {GFFFieldType.Struct, GFFFieldType.List}:
            return False

        raw_value = self._resolve_field_pointer(root_struct, self.value.resolve(memory))
        if raw_value is _INVALID:
            return False

        if field_type is GFFFieldType.LocalizedString:
            if not isinstance(raw_value, LocalizedStringDelta):
                return False
            original = navigated_struct.get_locstring(label)
            patched = _clone_locstring(original)
            if raw_value.apply(patched, memory, invalid_stringref=None):
                navigated_struct.set_locstring(label, patched)
                return True
            return False

        if field_type is GFFFieldType.Int8:
            parsed_value = _parse_char(raw_value, empty_value=_INVALID)
        elif field_type in _INTEGER_RANGES:
            parsed_value = _parse_int(raw_value, field_type, allow_negative=False)
        elif field_type in {GFFFieldType.Single, GFFFieldType.Double}:
            parsed_value = _parse_float(raw_value)
        elif field_type in {GFFFieldType.Vector3, GFFFieldType.Vector4}:
            parsed_value = _parse_vector(raw_value, field_type, field.value())
        else:
            parsed_value = _coerce_scalar(raw_value, field_type, existing=field.value(), new_field=False)

        if parsed_value is _INVALID:
            return False
        if field.value() == parsed_value:
            return False
        FIELD_TYPE_TO_SETTER[field_type](navigated_struct, label, parsed_value, memory)
        return True


class ModificationsGFF(PatcherModifications):
    def __init__(
        self,
        filename: str,
        replace: bool,  # noqa: FBT001
        modifiers: list[ModifyGFF] | None = None,
    ):
        super().__init__(filename, replace)
        self.modifiers = [] if modifiers is None else modifiers

    def patch_resource(
        self,
        source_gff: SOURCE_TYPES,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ) -> bytes | Literal[True]:
        reader = BinaryReader.from_auto(source_gff)
        try:
            source_bytes = reader.read_bytes(reader.remaining())
        finally:
            reader.close()
        gff: GFF = GFFBinaryReader(source_bytes).load()
        return bytes_gff(gff) if self.apply(gff, memory, logger, game) else source_bytes

    def apply(
        self,
        gff: GFF,
        memory: PatcherMemory,
        logger: PatchLogger,
        game: Game,
    ) -> bool:
        changed = False
        for modifier in self.modifiers:
            try:
                changed = modifier.apply(gff.root, memory, logger) or changed
            except Exception as exc:  # noqa: BLE001 - TSLPatcher continues after individual GFF operation failures.
                logger.add_error(f"Unable to apply GFF modifier [{getattr(modifier, 'identifier', '')}]: {exc}")
        return changed
