from __future__ import annotations

from typing import TYPE_CHECKING

from pykotor.resource.formats.ncs.ncs_data import NCS, NCSByteCode, NCSInstruction, NCSInstructionType, NCSInstructionTypeValue
from pykotor.resource.type import ResourceReader, ResourceWriter, autoclose

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


def _decode_ncs_string(data: bytes) -> str:
    """Decode NCS's one-byte string representation without losing undefined CP1252 bytes."""
    result: list[str] = []
    for byte in data:
        raw = bytes((byte,))
        try:
            result.append(raw.decode("windows-1252"))
        except UnicodeDecodeError:
            result.append(chr(byte))
    return "".join(result)


def _encode_ncs_string(value: str) -> bytes:
    """Encode an NCS string, preserving raw C1 byte values produced by NSS \\xNN escapes."""
    result = bytearray()
    for char in value:
        try:
            result.extend(char.encode("windows-1252"))
        except UnicodeEncodeError:
            codepoint = ord(char)
            if 0 <= codepoint <= 0xFF:
                result.append(codepoint)
            else:
                raise
    return bytes(result)


# NCS operands are big-endian. Keeping the fixed-width layouts in one table makes
# the reader, writer, and size calculation use the same binary contract.
_I32 = "i32"
_U32 = "u32"
_U16 = "u16"
_U8 = "u8"
_F32 = "f32"

_OPERAND_SIZES: dict[str, int] = {
    _I32: 4,
    _U32: 4,
    _U16: 2,
    _U8: 1,
    _F32: 4,
}

_FIXED_OPERAND_LAYOUTS: dict[NCSInstructionType, tuple[str, ...]] = {
    NCSInstructionType.CPDOWNSP: (_I32, _U16),
    NCSInstructionType.CPTOPSP: (_I32, _U16),
    NCSInstructionType.CPDOWNBP: (_I32, _U16),
    NCSInstructionType.CPTOPBP: (_I32, _U16),
    NCSInstructionType.CONSTI: (_I32,),
    NCSInstructionType.CONSTF: (_F32,),
    NCSInstructionType.CONSTO: (_U32,),
    NCSInstructionType.ACTION: (_U16, _U8),
    NCSInstructionType.MOVSP: (_I32,),
    NCSInstructionType.DESTRUCT: (_U16, _U16, _U16),
    NCSInstructionType.DECISP: (_I32,),
    NCSInstructionType.INCISP: (_I32,),
    NCSInstructionType.DECIBP: (_I32,),
    NCSInstructionType.INCIBP: (_I32,),
    NCSInstructionType.STORE_STATE: (_I32, _I32),
    NCSInstructionType.EQUALTT: (_U16,),
    NCSInstructionType.NEQUALTT: (_U16,),
}

_JUMP_INSTRUCTIONS = {
    NCSInstructionType.JMP,
    NCSInstructionType.JSR,
    NCSInstructionType.JZ,
    NCSInstructionType.JNZ,
}

_NO_OPERAND_INSTRUCTIONS = {
    NCSInstructionType.NOP,
    NCSInstructionType.RSADDI,
    NCSInstructionType.RSADDF,
    NCSInstructionType.RSADDS,
    NCSInstructionType.RSADDO,
    NCSInstructionType.RSADDEFF,
    NCSInstructionType.RSADDEVT,
    NCSInstructionType.RSADDLOC,
    NCSInstructionType.RSADDTAL,
    NCSInstructionType.LOGANDII,
    NCSInstructionType.LOGORII,
    NCSInstructionType.INCORII,
    NCSInstructionType.EXCORII,
    NCSInstructionType.BOOLANDII,
    NCSInstructionType.EQUALII,
    NCSInstructionType.EQUALFF,
    NCSInstructionType.EQUALOO,
    NCSInstructionType.EQUALEFFEFF,
    NCSInstructionType.EQUALEVTEVT,
    NCSInstructionType.EQUALLOCLOC,
    NCSInstructionType.EQUALTALTAL,
    NCSInstructionType.EQUALSS,
    NCSInstructionType.NEQUALII,
    NCSInstructionType.NEQUALFF,
    NCSInstructionType.NEQUALOO,
    NCSInstructionType.NEQUALEFFEFF,
    NCSInstructionType.NEQUALEVTEVT,
    NCSInstructionType.NEQUALLOCLOC,
    NCSInstructionType.NEQUALTALTAL,
    NCSInstructionType.NEQUALSS,
    NCSInstructionType.GEQII,
    NCSInstructionType.GEQFF,
    NCSInstructionType.GTII,
    NCSInstructionType.GTFF,
    NCSInstructionType.LTII,
    NCSInstructionType.LTFF,
    NCSInstructionType.LEQII,
    NCSInstructionType.LEQFF,
    NCSInstructionType.SHLEFTII,
    NCSInstructionType.SHRIGHTII,
    NCSInstructionType.USHRIGHTII,
    NCSInstructionType.ADDII,
    NCSInstructionType.ADDFF,
    NCSInstructionType.ADDFI,
    NCSInstructionType.ADDIF,
    NCSInstructionType.ADDSS,
    NCSInstructionType.ADDVV,
    NCSInstructionType.SUBII,
    NCSInstructionType.SUBFF,
    NCSInstructionType.SUBFI,
    NCSInstructionType.SUBIF,
    NCSInstructionType.SUBVV,
    NCSInstructionType.MULII,
    NCSInstructionType.MULFF,
    NCSInstructionType.MULFI,
    NCSInstructionType.MULIF,
    NCSInstructionType.MULFV,
    NCSInstructionType.MULVF,
    NCSInstructionType.DIVII,
    NCSInstructionType.DIVFF,
    NCSInstructionType.DIVFI,
    NCSInstructionType.DIVIF,
    NCSInstructionType.DIVFV,
    NCSInstructionType.DIVVF,
    NCSInstructionType.MODII,
    NCSInstructionType.NEGI,
    NCSInstructionType.NEGF,
    NCSInstructionType.COMPI,
    NCSInstructionType.RETN,
    NCSInstructionType.NOTI,
    NCSInstructionType.SAVEBP,
    NCSInstructionType.RESTOREBP,
}


class NCSBinaryReader(ResourceReader):
    def __init__(
        self,
        source: SOURCE_TYPES,
        offset: int = 0,
        size: int = 0,
    ):
        super().__init__(source, offset, size)
        self._ncs: NCS | None = None
        self._instructions: dict[int, NCSInstruction] = {}
        self._jumps: dict[NCSInstruction, int] = {}

    @autoclose
    def load(
        self,
        auto_close: bool = True,
    ) -> NCS:
        """Loads an NCS file from the reader.

        Args:
        ----
            auto_close: {Whether to automatically close the reader after loading}.

        Returns:
        -------
            NCS: The loaded NCS object

        Raises:
            ValueError - Corrupt NCS.
            OSError - some operating system issue occurred.

        Processing Logic:
        ----------------
            - Reads the file type and version headers
            - Reads each instruction from the file into a dictionary
            - Resolves jump offsets to reference the target instructions
            - Adds the instructions to the NCS object
            - Optionally closes the reader.
        """
        self._ncs = NCS()

        file_type = self._reader.read_string(4)
        file_version = self._reader.read_string(4)

        if file_type != "NCS ":
            msg = "The file type that was loaded is invalid."
            raise ValueError(msg)

        if file_version != "V1.0":
            msg = "The NCS version that was loaded is not supported."
            raise ValueError(msg)

        self._instructions = {}  # offset -> instruction

        self._reader.seek(13)
        while self._reader.remaining() > 0:
            offset = self._reader.position()
            self._instructions[offset] = self._read_instruction()

        for instruction, jumpToOffset in self._jumps.items():
            instruction.jump = self._instructions[jumpToOffset]

        self._ncs.instructions = list(self._instructions.values())

        return self._ncs

    def _read_fixed_operand(self, operand_type: str):
        """Read one fixed-width operand using the shared NCS layout definition."""
        if operand_type == _I32:
            return self._reader.read_int32(big=True)
        if operand_type == _U32:
            return self._reader.read_uint32(big=True)
        if operand_type == _U16:
            return self._reader.read_uint16(big=True)
        if operand_type == _U8:
            return self._reader.read_uint8()
        if operand_type == _F32:
            return self._reader.read_single(big=True)
        raise ValueError(f"Unsupported NCS operand type: {operand_type}")

    def _read_instruction(self) -> NCSInstruction:
        """Read one NCS instruction using the shared operand-layout table."""
        byte_code = NCSByteCode(self._reader.read_uint8())
        qualifier = self._reader.read_uint8()
        instruction_type = NCSInstructionType(NCSInstructionTypeValue(byte_code, qualifier))
        instruction = NCSInstruction(instruction_type)

        layout = _FIXED_OPERAND_LAYOUTS.get(instruction.ins_type)
        if layout is not None:
            instruction.args.extend(self._read_fixed_operand(operand_type) for operand_type in layout)
            return instruction

        if instruction.ins_type == NCSInstructionType.CONSTS:
            length = self._reader.read_uint16(big=True)
            instruction.args.append(_decode_ncs_string(self._reader.read_bytes(length)))
            return instruction

        if instruction.ins_type in _JUMP_INSTRUCTIONS:
            instruction_offset = self._reader.position() - 2
            relative_offset = self._reader.read_int32(big=True)
            self._jumps[instruction] = instruction_offset + relative_offset
            return instruction

        if instruction.ins_type in _NO_OPERAND_INSTRUCTIONS:
            return instruction

        msg = f"Tried to read unsupported instruction '{instruction.ins_type.name}' from NCS"
        raise ValueError(msg)



class NCSBinaryWriter(ResourceWriter):
    def __init__(
        self,
        ncs: NCS,
        target: TARGET_TYPES,
    ):
        super().__init__(target)
        self._ncs: NCS = ncs
        self._offsets: dict[NCSInstruction, int] = {}
        self._sizes: dict[NCSInstruction, int] = {}

    @autoclose
    def write(
        self,
        auto_close: bool = True,
    ):
        """Writes the NCS file.

        Args:
        ----
            auto_close (bool): Whether to automatically close the writer.

        Processing Logic:
        ----------------
            - Calculates offset and size for each instruction
            - Writes header with file type and total size
            - Writes each instruction using pre-calculated offset and size
            - Closes writer if auto_close is True.
        """
        offset = 13
        for instruction in self._ncs.instructions:
            self._sizes[instruction] = self.determine_size(instruction)
            self._offsets[instruction] = offset
            offset += self._sizes[instruction]

        self._writer.write_string("NCS ")
        self._writer.write_string("V1.0")
        self._writer.write_uint8(0x42)
        self._writer.write_uint32(offset, big=True)

        for instruction in self._ncs.instructions:
            self._write_instruction(instruction)

    def determine_size(self, instruction: NCSInstruction) -> int:
        """Return the encoded instruction size in bytes, including opcode and qualifier."""
        base_size = 2

        layout = _FIXED_OPERAND_LAYOUTS.get(instruction.ins_type)
        if layout is not None:
            return base_size + sum(_OPERAND_SIZES[operand_type] for operand_type in layout)

        if instruction.ins_type == NCSInstructionType.CONSTS:
            return base_size + 2 + len(_encode_ncs_string(instruction.args[0]))

        if instruction.ins_type in _JUMP_INSTRUCTIONS:
            return base_size + 4

        if instruction.ins_type in _NO_OPERAND_INSTRUCTIONS:
            return base_size

        msg = f"Tried to determine the size of unsupported instruction '{instruction.ins_type.name}'"
        raise ValueError(msg)

    def _write_fixed_operand(self, operand_type: str, value) -> None:
        """Write one fixed-width operand using the shared NCS layout definition."""
        if operand_type == _I32:
            self._writer.write_int32(value, big=True)
            return
        if operand_type == _U32:
            self._writer.write_uint32(value, big=True)
            return
        if operand_type == _U16:
            self._writer.write_uint16(value, big=True)
            return
        if operand_type == _U8:
            self._writer.write_uint8(value)
            return
        if operand_type == _F32:
            self._writer.write_single(value, big=True)
            return
        raise ValueError(f"Unsupported NCS operand type: {operand_type}")

    def _write_instruction(self, instruction: NCSInstruction) -> None:
        """Write one NCS instruction using the shared operand-layout table."""
        self._writer.write_uint8(int(instruction.ins_type.value.byte_code))
        self._writer.write_uint8(int(instruction.ins_type.value.qualifier))

        layout = _FIXED_OPERAND_LAYOUTS.get(instruction.ins_type)
        if layout is not None:
            if len(instruction.args) != len(layout):
                msg = (
                    f"Instruction '{instruction.ins_type.name}' expects {len(layout)} operands, "
                    f"but received {len(instruction.args)}."
                )
                raise ValueError(msg)
            for operand_type, value in zip(layout, instruction.args):
                self._write_fixed_operand(operand_type, value)
            return

        if instruction.ins_type == NCSInstructionType.CONSTS:
            data = _encode_ncs_string(instruction.args[0])
            if len(data) >= 1 << 16:
                raise ValueError("NCS string constant is too large for its 16-bit length prefix.")
            self._writer.write_uint16(len(data), big=True)
            self._writer.write_bytes(data)
            return

        if instruction.ins_type in _JUMP_INSTRUCTIONS:
            jump = instruction.jump
            if jump is None:
                raise ValueError(f"Instruction '{instruction.ins_type.name}' has no jump target.")
            if jump not in self._offsets:
                raise ValueError(f"Instruction '{instruction.ins_type.name}' targets an instruction outside this NCS object.")
            relative_offset = self._offsets[jump] - self._offsets[instruction]
            self._writer.write_int32(relative_offset, big=True)
            return

        if instruction.ins_type in _NO_OPERAND_INSTRUCTIONS:
            return

        msg = f"Tried to write unsupported instruction '{instruction.ins_type.name}' to NCS"
        raise ValueError(msg)
