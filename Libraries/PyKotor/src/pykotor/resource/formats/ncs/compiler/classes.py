"""NSS compiler AST and helpers: CompileError, expression/statement nodes, type helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
import struct
from utility.system.path import Path
from typing import TYPE_CHECKING, NamedTuple, cast

from pykotor.common.script import DataType
from pykotor.resource.formats.ncs import NCS, NCSInstruction, NCSInstructionType
from pykotor.tools.path import CaseAwarePath

if TYPE_CHECKING:
    from pykotor.common.script import ScriptConstant, ScriptFunction


def get_logical_equality_instruction(
    type1: DynamicDataType,
    type2: DynamicDataType,
) -> NCSInstructionType:
    if type1 == DataType.INT and type2 == DataType.INT:
        return NCSInstructionType.EQUALII
    if type1 == DataType.FLOAT and type2 == DataType.FLOAT:
        return NCSInstructionType.EQUALFF
    msg = f"Tried an unsupported comparison between '{type1}' '{type2}'."
    raise CompileError(msg)


class CompileError(Exception):
    """Base exception for NSS compilation errors.

    Provides detailed error messages to help debug script issues.

    References:
    ----------

    """

    def __init__(self, message: str, line_num: int | None = None, context: str | None = None):
        full_message = message
        if line_num is not None:
            full_message = f"Line {line_num}: {message}"
        if context:
            full_message = f"{full_message}\n  Context: {context}"
        super().__init__(full_message)
        self.line_num = line_num
        self.context = context


class EntryPointError(CompileError):
    """Raised when script has no valid entry point (main or StartingConditional)."""


class MissingIncludeError(CompileError):
    """Raised when a #include file cannot be found."""


DEFAULT_MAX_INCLUDE_DEPTH = 16
MAX_FUNCTION_PARAMETERS = 32


class IncludeContext:
    """Track BioWare-style include state for one root compilation.

    The original compiler treats the root script as file level 1. With the
    default limit of 16, at most 15 nested include files may therefore be
    active below the root at one time. Already-completed includes are skipped
    case-insensitively, while an include that is still active is recursive.
    """

    def __init__(self, max_depth: int = DEFAULT_MAX_INCLUDE_DEPTH):
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.max_depth = max_depth
        self.active: list[str] = []
        self.included: set[str] = set()

    @staticmethod
    def canonicalize(script_name: str) -> str:
        """Return the case-insensitive resource key used for include tracking."""
        return script_name.replace("\\", "/").casefold()

    def begin_include(self, script_name: str) -> str | None:
        """Enter an include, or return ``None`` when it was already included."""
        canonical_name = self.canonicalize(script_name)

        if canonical_name in self.active:
            cycle_start = self.active.index(canonical_name)
            cycle = [*self.active[cycle_start:], canonical_name]
            raise CompileError(f"Recursive include detected: {' -> '.join(cycle)}")

        if canonical_name in self.included:
            return None

        # Beamdog's m_nCompileFileLevel counts the root as level 1. Rejecting
        # here mirrors `m_nCompileFileLevel >= m_nMaxIncludeDepth` before the
        # next include file is entered.
        current_file_level = len(self.active) + 1
        if current_file_level >= self.max_depth:
            raise CompileError(
                f"Maximum include depth of {self.max_depth} file levels exceeded\n"
                "  The root script counts as the first file level"
            )

        self.active.append(canonical_name)
        return canonical_name

    def end_include(self, canonical_name: str, *, completed: bool) -> None:
        """Leave the active include and remember it only after successful parsing."""
        if not self.active or self.active[-1] != canonical_name:
            raise RuntimeError(
                "Internal compiler error: include stack was left in an inconsistent state"
            )
        self.active.pop()
        if completed:
            self.included.add(canonical_name)


class ConstantValue(NamedTuple):
    """A BioWare-style compile-time NSS constant."""

    datatype: DataType
    value: int | float | str


@dataclass(frozen=True)
class SourceOrigin:
    """Source position of one top-level declaration after include expansion.

    ``file_level`` mirrors BioWare's compiler file level (root script = 1,
    includes >= 2). ``source_order`` is the depth-first declaration order after
    includes have been expanded at their source position.
    """

    file_level: int
    source_order: int


@dataclass(frozen=True)
class CompileTimeConstantSymbol:
    """One predefined or source-declared compile-time constant."""

    value: ConstantValue
    origin: SourceOrigin | None = None

    def is_visible_at(self, source_origin: SourceOrigin | None) -> bool:
        """Return whether this constant is visible from ``source_origin``."""
        if self.origin is None or source_origin is None:
            return True
        return self.origin.source_order <= source_origin.source_order


def _int32(value: int) -> int:
    """Normalize an integer to the signed 32-bit representation used by NCS."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def _float32(value: float) -> float:
    """Round to the 32-bit IEEE-754 representation used by NCS CONSTF."""
    try:
        return struct.unpack(">f", struct.pack(">f", float(value)))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _emit_constant(ncs: NCS, constant: ConstantValue) -> DynamicDataType:
    if constant.datatype == DataType.INT:
        ncs.add(NCSInstructionType.CONSTI, args=[_int32(int(constant.value))])
    elif constant.datatype == DataType.FLOAT:
        ncs.add(NCSInstructionType.CONSTF, args=[_float32(float(constant.value))])
    elif constant.datatype == DataType.STRING:
        ncs.add(NCSInstructionType.CONSTS, args=[str(constant.value)])
    else:
        raise CompileError(f"Unsupported compile-time constant type: {constant.datatype.name}")
    return DynamicDataType(constant.datatype)


def _c_int_div(left: int, right: int) -> int:
    quotient = abs(left) // abs(right)
    return -quotient if (left < 0) != (right < 0) else quotient


def _fold_unary_constant(
    instruction: NCSInstructionType,
    operand: ConstantValue,
) -> ConstantValue | None:
    if operand.datatype == DataType.INT:
        value = _int32(int(operand.value))
        if instruction == NCSInstructionType.NEGI:
            return ConstantValue(DataType.INT, _int32(-value))
        if instruction == NCSInstructionType.COMPI:
            return ConstantValue(DataType.INT, _int32(~value))
        if instruction == NCSInstructionType.NOTI:
            return ConstantValue(DataType.INT, int(not value))
    elif operand.datatype == DataType.FLOAT and instruction == NCSInstructionType.NEGF:
        return ConstantValue(DataType.FLOAT, _float32(-float(operand.value)))
    return None


def _fold_binary_constant(
    mapping: BinaryOperatorMapping,
    left: ConstantValue,
    right: ConstantValue,
) -> ConstantValue | None:
    # BioWare's constant folder folds same-type operands only (except short-circuit
    # logical expressions, handled before this helper is called).
    if left.datatype != right.datatype:
        return None
    if left.datatype != mapping.lhs or right.datatype != mapping.rhs:
        return None

    op = mapping.instruction
    if left.datatype == DataType.INT:
        lhs = _int32(int(left.value))
        rhs = _int32(int(right.value))
        if op == NCSInstructionType.LOGORII:
            result = int(bool(lhs) or bool(rhs))
        elif op == NCSInstructionType.LOGANDII:
            result = int(bool(lhs) and bool(rhs))
        elif op == NCSInstructionType.INCORII:
            result = lhs | rhs
        elif op == NCSInstructionType.EXCORII:
            result = lhs ^ rhs
        elif op == NCSInstructionType.BOOLANDII:
            result = lhs & rhs
        elif op == NCSInstructionType.EQUALII:
            result = int(lhs == rhs)
        elif op == NCSInstructionType.NEQUALII:
            result = int(lhs != rhs)
        elif op == NCSInstructionType.GEQII:
            result = int(lhs >= rhs)
        elif op == NCSInstructionType.GTII:
            result = int(lhs > rhs)
        elif op == NCSInstructionType.LTII:
            result = int(lhs < rhs)
        elif op == NCSInstructionType.LEQII:
            result = int(lhs <= rhs)
        elif op == NCSInstructionType.SHLEFTII:
            if not 0 <= rhs < 32:
                return None
            result = lhs << rhs
        elif op == NCSInstructionType.SHRIGHTII:
            if not 0 <= rhs < 32:
                return None
            result = lhs >> rhs
        elif op == NCSInstructionType.USHRIGHTII:
            if not 0 <= rhs < 32:
                return None
            result = (lhs & 0xFFFFFFFF) >> rhs
        elif op == NCSInstructionType.ADDII:
            result = lhs + rhs
        elif op == NCSInstructionType.SUBII:
            result = lhs - rhs
        elif op == NCSInstructionType.MULII:
            result = lhs * rhs
        elif op in (NCSInstructionType.DIVII, NCSInstructionType.MODII):
            if rhs == 0 or (lhs == -0x80000000 and rhs == -1):
                return None
            quotient = _c_int_div(lhs, rhs)
            result = quotient if op == NCSInstructionType.DIVII else lhs - quotient * rhs
        else:
            return None
        return ConstantValue(DataType.INT, _int32(result))

    if left.datatype == DataType.FLOAT:
        lhs = _float32(float(left.value))
        rhs = _float32(float(right.value))
        if op == NCSInstructionType.ADDFF:
            return ConstantValue(DataType.FLOAT, _float32(lhs + rhs))
        if op == NCSInstructionType.SUBFF:
            return ConstantValue(DataType.FLOAT, _float32(lhs - rhs))
        if op == NCSInstructionType.MULFF:
            return ConstantValue(DataType.FLOAT, _float32(lhs * rhs))
        if op == NCSInstructionType.DIVFF:
            if rhs == 0.0:
                if lhs == 0.0:
                    result = math.nan
                else:
                    result = math.copysign(math.inf, lhs * math.copysign(1.0, rhs))
            else:
                result = lhs / rhs
            return ConstantValue(DataType.FLOAT, _float32(result))
        comparisons = {
            NCSInstructionType.EQUALFF: lhs == rhs,
            NCSInstructionType.NEQUALFF: lhs != rhs,
            NCSInstructionType.GEQFF: lhs >= rhs,
            NCSInstructionType.GTFF: lhs > rhs,
            NCSInstructionType.LTFF: lhs < rhs,
            NCSInstructionType.LEQFF: lhs <= rhs,
        }
        if op in comparisons:
            return ConstantValue(DataType.INT, int(comparisons[op]))
        return None

    if left.datatype == DataType.STRING:
        lhs = str(left.value)
        rhs = str(right.value)
        if op == NCSInstructionType.ADDSS:
            return ConstantValue(DataType.STRING, lhs + rhs)
        if op == NCSInstructionType.EQUALSS:
            return ConstantValue(DataType.INT, int(lhs == rhs))
        if op == NCSInstructionType.NEQUALSS:
            return ConstantValue(DataType.INT, int(lhs != rhs))
    return None


class TopLevelObject(ABC):
    source_origin: SourceOrigin | None = None

    def set_source_origin(self, origin: SourceOrigin) -> None:
        """Attach the declaration's explicit position in the expanded source."""
        self.source_origin = origin

    def require_source_origin(self) -> SourceOrigin:
        """Return source metadata after include expansion has annotated the AST."""
        if self.source_origin is None:
            raise ValueError(
                "Internal compiler error: top-level object has no source origin"
            )
        return self.source_origin

    def register(self, root: CodeRoot) -> None:
        """Register semantic symbols without emitting NCS bytecode."""

    @abstractmethod
    def compile(self, ncs: NCS, root: CodeRoot):  # noqa: A003
        """Emit NCS bytecode after the registration pass has completed."""
        ...


class GlobalVariableInitialization(TopLevelObject):
    def __init__(
        self,
        identifier: Identifier,
        data_type: DynamicDataType,
        value: Expression,
        is_const: bool = False,
    ):
        super().__init__()
        self.identifier: Identifier = identifier
        self.data_type: DynamicDataType = data_type
        self.expression: Expression = value
        self.is_const: bool = is_const

    def register(self, root: CodeRoot) -> None:
        root.register_global(
            self.identifier,
            self.data_type,
            is_const=self.is_const,
            expression=self.expression if self.is_const else None,
            origin=self.require_source_origin(),
        )

    def compile(self, ncs: NCS, root: CodeRoot):
        if self.is_const:
            return

        # Semantic registration and VM storage allocation are intentionally separate.
        # Only globals whose storage has actually been emitted participate in BP/SP
        # offset lookup during the global-initializer phase.
        declaration = GlobalVariableDeclaration(self.identifier, self.data_type)
        declaration._emit_storage(ncs, root)

        block = CodeBlock(
            CompilationContext(
                SemanticContext(source_origin=self.require_source_origin())
            )
        )
        expression_type = self.expression.compile(ncs, root, block)
        if expression_type != self.data_type:
            msg = (
                f"Type mismatch in initialization of global variable '{self.identifier}'\n"
                f"  Declared type: {self.data_type.builtin.name}\n"
                f"  Initializer type: {expression_type.builtin.name}"
            )
            raise CompileError(msg)

        scoped = root.get_scoped(self.identifier, root)
        # Global storage resides on the stack before base pointer is saved, so use stack-pointer-relative copy.
        stack_index = scoped.offset - scoped.datatype.size(root)
        ncs.instructions.append(
            NCSInstruction(
                NCSInstructionType.CPDOWNSP,
                [stack_index, scoped.datatype.size(root)],
            ),
        )
        # Remove the initializer value from the stack and mirror that in the explicit stack context.
        initializer_size = scoped.datatype.size(root)
        ncs.add(NCSInstructionType.MOVSP, args=[-initializer_size])
        block.context.stack.consume(initializer_size)


class GlobalVariableDeclaration(TopLevelObject):
    def __init__(self, identifier: Identifier, data_type: DynamicDataType, is_const: bool = False):
        super().__init__()
        self.identifier: Identifier = identifier
        self.data_type: DynamicDataType = data_type
        self.is_const: bool = is_const

    def register(self, root: CodeRoot) -> None:
        root.register_global(
            self.identifier,
            self.data_type,
            is_const=self.is_const,
            expression=None,
            origin=self.require_source_origin(),
        )

    def compile(self, ncs: NCS, root: CodeRoot):  # noqa: A003
        if self.is_const:
            return
        self._emit_storage(ncs, root)

    def _emit_storage(self, ncs: NCS, root: CodeRoot) -> None:
        if self.data_type.builtin == DataType.INT:
            ncs.add(NCSInstructionType.RSADDI)
        elif self.data_type.builtin == DataType.FLOAT:
            ncs.add(NCSInstructionType.RSADDF)
        elif self.data_type.builtin == DataType.STRING:
            ncs.add(NCSInstructionType.RSADDS)
        elif self.data_type.builtin == DataType.OBJECT:
            ncs.add(NCSInstructionType.RSADDO)
        elif self.data_type.builtin == DataType.EVENT:
            ncs.add(NCSInstructionType.RSADDEVT)
        elif self.data_type.builtin == DataType.LOCATION:
            ncs.add(NCSInstructionType.RSADDLOC)
        elif self.data_type.builtin == DataType.TALENT:
            ncs.add(NCSInstructionType.RSADDTAL)
        elif self.data_type.builtin == DataType.EFFECT:
            ncs.add(NCSInstructionType.RSADDEFF)
        elif self.data_type.builtin == DataType.VECTOR:
            ncs.add(NCSInstructionType.RSADDF)
            ncs.add(NCSInstructionType.RSADDF)
            ncs.add(NCSInstructionType.RSADDF)
        elif self.data_type.builtin == DataType.STRUCT:
            struct_name = self.data_type._struct  # noqa: SLF001
            if struct_name is not None and struct_name in root.struct_map:
                root.struct_map[struct_name].initialize(ncs, root)
            else:
                msg = f"Unknown struct type for variable '{self.identifier}'"
                raise CompileError(msg)
        elif self.data_type.builtin == DataType.VOID:
            msg = f"Cannot declare variable '{self.identifier}' with void type\n  void can only be used as a function return type"
            raise CompileError(msg)
        else:
            msg = f"Unsupported type '{self.data_type.builtin.name}' for global variable '{self.identifier}'\n  This may indicate a compiler bug or unsupported type"
            raise CompileError(msg)

        root.allocate_global(self.identifier, self.data_type, is_const=self.is_const)


class Identifier:
    def __init__(self, label: str):
        self.label: str = label

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Identifier):
            return self.label == other.label
        if isinstance(other, str):
            return self.label == other
        return NotImplemented  # type: ignore[no-any-return]

    def __str__(self):
        return self.label

    def __hash__(self):
        return hash(self.label)


class ControlKeyword(Enum):
    BREAK = "break"
    CASE = "control"
    DEFAULT = "default"
    DO = "do"
    ELSE = "else"
    SWITCH = "switch"
    WHILE = "while"
    FOR = "for"
    IF = "if"
    RETURN = "return"


class Operator(Enum):
    ADDITION = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULUS = "%"
    NOT = "!"
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    AND = "&&"
    OR = "||"
    BITWISE_AND = "&"
    BITWISE_OR = "|"
    BITWISE_XOR = "^"
    BITWISE_LEFT = "<<"
    BITWISE_RIGHT = ">>"
    ONES_COMPLEMENT = "~"


class SemanticContext:
    """Function-level semantic state shared by every block in one compilation unit."""

    def __init__(
        self,
        function_name: str | None = None,
        return_type: DynamicDataType | None = None,
        source_origin: SourceOrigin | None = None,
    ):
        self.function_name = function_name
        self.return_type = return_type
        self.source_origin = source_origin


class StackContext:
    """Tracks transient VM stack bytes produced while evaluating expressions."""

    def __init__(self):
        self.temporary_bytes = 0

    def snapshot(self) -> int:
        return self.temporary_bytes

    def restore(self, temporary_bytes: int) -> None:
        if temporary_bytes < 0:
            raise ValueError("Internal compiler error: negative temporary stack depth")
        self.temporary_bytes = temporary_bytes

    def push(self, size: int) -> None:
        if size < 0:
            raise ValueError("Internal compiler error: cannot push a negative stack size")
        self.temporary_bytes += size

    def consume(self, size: int) -> None:
        if size < 0 or size > self.temporary_bytes:
            raise ValueError(
                "Internal compiler error: temporary stack underflow "
                f"({size} bytes requested, {self.temporary_bytes} available)"
            )
        self.temporary_bytes -= size

    def replace(self, consumed: int, produced: int) -> None:
        self.consume(consumed)
        self.push(produced)


class ControlFlowTarget:
    """One lexical break/continue target and the VM stack depth required at it."""

    def __init__(
        self,
        kind: ControlKeyword,
        break_instruction: NCSInstruction,
        break_stack_depth: int,
        continue_instruction: NCSInstruction | None = None,
        continue_stack_depth: int | None = None,
    ):
        self.kind = kind
        self.break_instruction = break_instruction
        self.break_stack_depth = break_stack_depth
        self.continue_instruction = continue_instruction
        self.continue_stack_depth = continue_stack_depth


class ControlFlowContext:
    """Lexical stack of loop/switch targets used by break and continue."""

    def __init__(self):
        self.targets: list[ControlFlowTarget] = []

    def push(self, target: ControlFlowTarget) -> None:
        self.targets.append(target)

    def pop(self, target: ControlFlowTarget) -> None:
        if not self.targets or self.targets[-1] is not target:
            raise ValueError("Internal compiler error: control-flow target stack mismatch")
        self.targets.pop()

    def break_target(self) -> ControlFlowTarget | None:
        return self.targets[-1] if self.targets else None

    def continue_target(self) -> ControlFlowTarget | None:
        return next(
            (target for target in reversed(self.targets) if target.continue_instruction is not None),
            None,
        )


class CompilationContext:
    """Explicit semantic, transient-stack, and control-flow compilation state."""

    def __init__(
        self,
        semantic: SemanticContext | None = None,
    ):
        self.semantic = semantic or SemanticContext()
        self.stack = StackContext()
        self.control = ControlFlowContext()


class OperatorMapping(NamedTuple):
    unary: list[UnaryOperatorMapping]
    binary: list[BinaryOperatorMapping]


class BinaryOperatorMapping:
    def __init__(
        self,
        instruction: NCSInstructionType,
        result: DataType,
        lhs: DataType,
        rhs: DataType,
    ):
        self.instruction: NCSInstructionType = instruction
        self.result: DataType = result
        self.lhs: DataType = lhs
        self.rhs: DataType = rhs

    def __repr__(self):
        return f"{self.__class__.__name__}(instruction={self.instruction!r}, result={self.result!r}, lhs={self.lhs!r}, rhs={self.rhs!r})"


class UnaryOperatorMapping:
    def __init__(self, instruction: NCSInstructionType, rhs: DataType):
        self.instruction: NCSInstructionType = instruction
        self.rhs: DataType = rhs




def _format_function_type(datatype: DynamicDataType) -> str:
    """Return a source-like type name for function-signature diagnostics."""
    if datatype.builtin == DataType.STRUCT:
        return f"struct {datatype._struct}"  # noqa: SLF001
    return datatype.builtin.name.lower()


@dataclass(frozen=True)
class FunctionSignature:
    """The parts of a user-function declaration that define its type signature."""

    return_type: DynamicDataType
    parameter_types: tuple[DynamicDataType, ...]

    @classmethod
    def from_function(
        cls,
        function: FunctionForwardDeclaration | FunctionDefinition,
    ) -> FunctionSignature:
        return cls(
            return_type=function.return_type,
            parameter_types=tuple(parameter.data_type for parameter in function.parameters),
        )


@dataclass(frozen=True)
class EngineFunctionReference:
    """One predefined engine routine and its ACTION opcode routine index."""

    routine_id: int
    function: ScriptFunction


@dataclass
class FunctionSymbol:
    """Semantic state for one user-defined function name.

    A declaration contributes only a signature.  An implementation contributes the
    executable entry label.  Keeping those concepts separate prevents a prototype
    from accidentally becoming a valid JSR destination.
    """

    name: str
    signature: FunctionSignature
    parameters: tuple[FunctionDefinitionParam, ...]
    implementation: FunctionDefinition | None = None
    entry_instruction: NCSInstruction | None = None
    referenced: bool = False
    unresolved_target: NCSInstruction | None = None
    declaration_origins: list[SourceOrigin] = field(default_factory=list)
    implementation_origin: SourceOrigin | None = None

    def add_declaration_origin(self, origin: SourceOrigin) -> None:
        """Record one prototype/definition site for visibility and entry selection."""
        self.declaration_origins.append(origin)

    def is_visible_at(self, source_origin: SourceOrigin | None) -> bool:
        """Return whether at least one declaration precedes this source location."""
        if source_origin is None:
            return True
        return any(
            origin.source_order <= source_origin.source_order
            for origin in self.declaration_origins
        )

    @property
    def has_root_declaration(self) -> bool:
        """Whether ``main``/``StartingConditional`` was declared in the root file."""
        return any(origin.file_level == 1 for origin in self.declaration_origins)

    def require_entry_instruction(self) -> NCSInstruction:
        """Return the implementation entry label, or fail on invalid compiler state."""
        if self.entry_instruction is None:
            raise ValueError(
                f"Internal compiler error: function '{self.name}' has no executable entry label"
            )
        return self.entry_instruction

    def note_reference(self) -> None:
        """Record a semantic reference to this function without emitting a call."""
        self.referenced = True

    def mark_referenced(self) -> NCSInstruction:
        """Record a call and return the label used by its temporary JSR.

        All top-level symbols are registered before emission, so a real implementation
        already has an entry label here.  Prototype-only calls use a non-emitted
        placeholder solely so bytecode emission can finish and report all unresolved
        calls in one final validation step.
        """
        self.note_reference()
        if self.entry_instruction is not None:
            return self.entry_instruction
        if self.unresolved_target is None:
            self.unresolved_target = NCSInstruction(NCSInstructionType.NOP)
        return self.unresolved_target


class GetScopedResult(NamedTuple):
    is_global: bool
    datatype: DynamicDataType
    offset: int
    is_const: bool = False


class Struct:
    def __init__(self, identifier: Identifier, members: list[StructMember]):
        self.identifier: Identifier = identifier
        self.members: list[StructMember] = members
        self._cached_size: int | None = None  # Cache size for performance

    def initialize(self, ncs: NCS, root: CodeRoot):
        for member in self.members:
            member.initialize(ncs, root)

    def size(self, root: CodeRoot) -> int:
        """Calculate struct size with caching for performance."""
        if self._cached_size is None:
            self._cached_size = sum(member.size(root) for member in self.members)
        return self._cached_size

    def child_offset(self, root: CodeRoot, identifier: Identifier) -> int:
        size = 0
        for member in self.members:
            if member.identifier == identifier:
                break
            size += member.size(root)
        else:
            # Provide helpful error with available members
            available = [m.identifier.label for m in self.members]
            msg = f"Unknown member '{identifier}' in struct '{self.identifier}'\n  Available members: {', '.join(available)}"
            raise CompileError(msg)
        return size

    def child_type(self, root: CodeRoot, identifier: Identifier) -> DynamicDataType:
        for member in self.members:
            if member.identifier == identifier:
                return member.datatype
        available = [m.identifier.label for m in self.members]
        msg = f"Member '{identifier}' not found in struct '{self.identifier}'\n  Available members: {', '.join(available)}"
        raise CompileError(msg)


class StructMember:
    def __init__(self, datatype: DynamicDataType, identifier: Identifier):
        self.datatype: DynamicDataType = datatype
        self.identifier: Identifier = identifier

    def initialize(self, ncs: NCS, root: CodeRoot):
        if self.datatype.builtin == DataType.INT:
            ncs.add(NCSInstructionType.RSADDI, args=[])
        elif self.datatype.builtin == DataType.FLOAT:
            ncs.add(NCSInstructionType.RSADDF, args=[])
        elif self.datatype.builtin == DataType.STRING:
            ncs.add(NCSInstructionType.RSADDS, args=[])
        elif self.datatype.builtin == DataType.OBJECT:
            ncs.add(NCSInstructionType.RSADDO, args=[])
        elif self.datatype.builtin == DataType.EVENT:
            ncs.add(NCSInstructionType.RSADDEVT, args=[])
        elif self.datatype.builtin == DataType.LOCATION:
            ncs.add(NCSInstructionType.RSADDLOC, args=[])
        elif self.datatype.builtin == DataType.TALENT:
            ncs.add(NCSInstructionType.RSADDTAL, args=[])
        elif self.datatype.builtin == DataType.EFFECT:
            ncs.add(NCSInstructionType.RSADDEFF, args=[])
        elif self.datatype.builtin == DataType.VECTOR:
            ncs.add(NCSInstructionType.RSADDF, args=[])
            ncs.add(NCSInstructionType.RSADDF, args=[])
            ncs.add(NCSInstructionType.RSADDF, args=[])
        elif self.datatype.builtin == DataType.STRUCT:
            # Use the struct type name from datatype, not the member name
            struct_type_name = self.datatype._struct
            if struct_type_name is None:
                msg = f"Struct member '{self.identifier.label}' has no struct type name"
                raise CompileError(msg)
            if struct_type_name not in root.struct_map:
                msg = (
                    f"Unknown struct type '{struct_type_name}' for member '{self.identifier.label}'"
                )
                raise CompileError(msg)
            root.struct_map[struct_type_name].initialize(ncs, root)
        else:
            msg = (
                f"Unsupported struct member type: {self.datatype.builtin.name}\n"
                f"  Member: {self.identifier}\n"
                f"  Supported types: int, float, string, object, vector, event, effect, location, talent, struct"
            )
            raise CompileError(msg)

    def size(self, root: CodeRoot) -> int:
        return self.datatype.size(root)


class CodeRoot:
    """Root compilation context for NSS compilation.

    Manages global scope, function definitions, constants, and compilation state.
    Provides symbol resolution and type checking during NSS to NCS compilation.

    References:
    ----------

    """

    def __init__(
        self,
        constants: list[ScriptConstant],
        functions: list[ScriptFunction],
        library_lookup: list[str] | list[Path] | list[Path | str] | str | Path | None,
        library: dict[str, bytes],
        max_include_depth: int = DEFAULT_MAX_INCLUDE_DEPTH,
    ):
        self.objects: list[TopLevelObject] = []

        self.library: dict[str, bytes] = library
        self.include_context = IncludeContext(max_include_depth)
        self.functions: list[ScriptFunction] = functions
        self._engine_function_map: dict[str, EngineFunctionReference] = {}
        for routine_id, function in enumerate(functions):
            if function.name in self._engine_function_map:
                raise ValueError(
                    f"Duplicate engine function metadata for '{function.name}'"
                )
            self._engine_function_map[function.name] = EngineFunctionReference(
                routine_id,
                function,
            )
        self.constants: list[ScriptConstant] = constants
        self.library_lookup: list[Path] = []
        if library_lookup:
            if not isinstance(library_lookup, list):
                library_lookup = [library_lookup]
            normalized: list[Path] = []
            for item in library_lookup:
                path_obj = CaseAwarePath(item)
                normalized.append(path_obj)
            self.library_lookup = normalized

        self.function_map: dict[str, FunctionSymbol] = {}
        self._global_scope: list[ScopedValue] = []
        self._compile_time_constants: dict[str, CompileTimeConstantSymbol] = {}
        for constant in constants:
            if constant.datatype == DataType.INT:
                value: int | float | str = _int32(int(constant.value))
            elif constant.datatype == DataType.FLOAT:
                value = _float32(float(constant.value))
            elif constant.datatype == DataType.STRING:
                value = str(constant.value)
            else:
                continue
            self._compile_time_constants[constant.name] = CompileTimeConstantSymbol(
                ConstantValue(constant.datatype, value)
            )
        self.struct_map: dict[str, Struct] = {}
        # Semantic symbol registration is separate from emitted VM storage.
        # ``_registered_globals`` detects declarations before any bytecode exists,
        # while ``_global_scope`` contains only globals whose runtime slots have
        # actually been emitted.
        self._registered_globals: dict[str, ScopedValue] = {}
        self._symbols_registered = False
        self._next_source_order = 0

    def expand_includes(self) -> None:
        """Expand includes and annotate every declaration with its source origin."""
        self._next_source_order = 0
        self.objects = self._expand_include_objects(self.objects, file_level=1)

    def _expand_include_objects(
        self,
        objects: list[TopLevelObject],
        *,
        file_level: int,
    ) -> list[TopLevelObject]:
        expanded: list[TopLevelObject] = []
        for obj in objects:
            if isinstance(obj, IncludeScript):
                expanded.extend(obj.expand(self, self.include_context))
            else:
                obj.set_source_origin(SourceOrigin(file_level, self._next_source_order))
                self._next_source_order += 1
                expanded.append(obj)
        return expanded

    def register_symbols(self) -> None:
        """Build and validate the complete top-level symbol table.

        Registration deliberately performs no NCS emission.  Struct names are
        collected first so every later signature/declaration can be validated,
        globals/constants are then registered in source order (preserving constant
        dependency semantics), and function signatures are registered last. Function
        registration is complete for code layout, while ``SourceOrigin`` still
        controls whether a function was visible at any particular source location.
        """
        if self._symbols_registered:
            return

        struct_definitions = [obj for obj in self.objects if isinstance(obj, StructDefinition)]
        for definition in struct_definitions:
            definition.register(self)
        self._validate_struct_definitions()

        for obj in self.objects:
            if isinstance(obj, (GlobalVariableDeclaration, GlobalVariableInitialization)):
                obj.register(self)

        for obj in self.objects:
            if isinstance(obj, (FunctionForwardDeclaration, FunctionDefinition)):
                obj.register(self)

        self._symbols_registered = True

    def register_struct(self, definition: StructDefinition) -> None:
        name = definition.identifier.label
        if name in self.struct_map:
            raise CompileError(f"Struct '{name}' is already defined")
        if not definition.members:
            raise CompileError(
                f"Struct '{name}' cannot be empty\n  Structs must have at least one member"
            )
        self.struct_map[name] = Struct(definition.identifier, definition.members)

    def _validate_struct_definitions(self) -> None:
        """Validate member names/types and reject recursive by-value layouts."""
        for name, struct_type in self.struct_map.items():
            seen_members: set[str] = set()
            for member in struct_type.members:
                member_name = member.identifier.label
                supported_member_types = {
                    DataType.INT,
                    DataType.FLOAT,
                    DataType.STRING,
                    DataType.OBJECT,
                    DataType.VECTOR,
                    DataType.EVENT,
                    DataType.EFFECT,
                    DataType.LOCATION,
                    DataType.TALENT,
                    DataType.STRUCT,
                }
                if member.datatype.builtin not in supported_member_types:
                    raise CompileError(
                        f"Unsupported type '{member.datatype.builtin.name.lower()}' for "
                        f"member '{member_name}' of struct '{name}'"
                    )
                if member_name in seen_members:
                    raise CompileError(
                        f"Member '{member_name}' is declared more than once in struct '{name}'"
                    )
                seen_members.add(member_name)
                self.validate_data_type(
                    member.datatype,
                    context=f"member '{member_name}' of struct '{name}'",
                    allow_void=False,
                )

        visiting: list[str] = []
        validated: set[str] = set()

        def validate_layout(name: str) -> int:
            if name in validated:
                cached = self.struct_map[name]._cached_size  # noqa: SLF001
                assert cached is not None
                return cached
            if name in visiting:
                cycle = " -> ".join([*visiting[visiting.index(name):], name])
                raise CompileError(
                    f"Recursive struct layout is not allowed: {cycle}\n"
                    "  Struct members are stored by value and must have a finite size"
                )

            visiting.append(name)
            total_size = 0
            for member in self.struct_map[name].members:
                if member.datatype.builtin == DataType.STRUCT:
                    nested_name = member.datatype._struct  # noqa: SLF001
                    assert nested_name is not None
                    total_size += validate_layout(nested_name)
                else:
                    total_size += member.datatype.builtin.size()
            visiting.pop()
            self.struct_map[name]._cached_size = total_size  # noqa: SLF001
            validated.add(name)
            return total_size

        for name in self.struct_map:
            validate_layout(name)

    def validate_data_type(
        self,
        datatype: DynamicDataType,
        *,
        context: str,
        allow_void: bool,
    ) -> None:
        if datatype.builtin == DataType.VOID:
            if not allow_void:
                raise CompileError(f"Invalid void type for {context}")
            return
        if datatype.builtin == DataType.STRUCT:
            struct_name = datatype._struct  # noqa: SLF001
            if not struct_name or struct_name not in self.struct_map:
                raise CompileError(f"Unknown struct type '{struct_name}' for {context}")

    def register_global(
        self,
        identifier: Identifier,
        datatype: DynamicDataType,
        *,
        is_const: bool,
        expression: Expression | None,
        origin: SourceOrigin,
    ) -> None:
        name = identifier.label
        if name in self._registered_globals or name in self._compile_time_constants:
            raise CompileError(f"Identifier '{identifier}' is already declared")

        self.validate_data_type(
            datatype,
            context=f"global variable '{name}'",
            allow_void=False,
        )
        self._registered_globals[name] = ScopedValue(identifier, datatype, is_const)
        if is_const:
            self._define_compile_time_constant(identifier, datatype, expression, origin)

    def register_function(
        self,
        function: FunctionForwardDeclaration | FunctionDefinition,
    ) -> None:
        """Register a function declaration or implementation without emitting code."""
        name = function.identifier.label
        origin = function.require_source_origin()
        if name in self._engine_function_map:
            raise CompileError(
                f"Function '{name}' conflicts with a predefined engine function\n"
                "  User-defined functions cannot reuse engine function names"
            )

        self.validate_data_type(
            function.return_type,
            context=f"return type of function '{name}'",
            allow_void=True,
        )

        if len(function.parameters) > MAX_FUNCTION_PARAMETERS:
            raise CompileError(
                f"Function '{name}' has too many parameters\n"
                f"  Maximum supported: {MAX_FUNCTION_PARAMETERS}\n"
                f"  Declared: {len(function.parameters)}"
            )

        seen_parameters: set[str] = set()
        for parameter in function.parameters:
            parameter_name = parameter.identifier.label
            if parameter_name in seen_parameters:
                raise CompileError(
                    f"Parameter '{parameter_name}' is declared more than once in function '{name}'"
                )
            seen_parameters.add(parameter_name)
            self.validate_data_type(
                parameter.data_type,
                context=f"parameter '{parameter_name}' of function '{name}'",
                allow_void=False,
            )

        _validate_and_fold_default_parameters(function.parameters, self, name, origin)

        signature = FunctionSignature.from_function(function)
        existing = self.function_map.get(name)
        is_implementation = isinstance(function, FunctionDefinition)

        if existing is None:
            self.function_map[name] = FunctionSymbol(
                name=name,
                signature=signature,
                parameters=tuple(function.parameters),
                implementation=function if is_implementation else None,
                entry_instruction=(
                    NCSInstruction(NCSInstructionType.NOP) if is_implementation else None
                ),
                declaration_origins=[origin],
                implementation_origin=origin if is_implementation else None,
            )
            return

        if existing.signature != signature:
            self._raise_function_signature_mismatch(name, existing.signature, signature)

        existing.add_declaration_origin(origin)

        # BioWare accepts repeated identical declarations.  A declaration after an
        # implementation is equally harmless because it adds no executable code.
        if not is_implementation:
            return

        if existing.implementation is not None:
            raise CompileError(
                f"Function '{name}' is already defined\n"
                "  Cannot redefine a function that already has an implementation"
            )

        existing.implementation = function
        existing.entry_instruction = NCSInstruction(NCSInstructionType.NOP)
        existing.implementation_origin = origin

    @staticmethod
    def _raise_function_signature_mismatch(
        name: str,
        existing: FunctionSignature,
        incoming: FunctionSignature,
    ) -> None:
        details: list[str] = []
        if existing.return_type != incoming.return_type:
            details.append(
                f"return type is {_format_function_type(existing.return_type)} vs "
                f"{_format_function_type(incoming.return_type)}"
            )
        if len(existing.parameter_types) != len(incoming.parameter_types):
            details.append(
                f"parameter count is {len(existing.parameter_types)} vs "
                f"{len(incoming.parameter_types)}"
            )
        else:
            for index, (old_type, new_type) in enumerate(
                zip(existing.parameter_types, incoming.parameter_types),
                start=1,
            ):
                if old_type != new_type:
                    details.append(
                        f"parameter {index} is {_format_function_type(old_type)} vs "
                        f"{_format_function_type(new_type)}"
                    )

        detail_text = "; ".join(details) or "signatures differ"
        raise CompileError(
            f"Function '{name}' declaration does not match the existing signature\n"
            f"  {detail_text}"
        )

    def validate_referenced_functions(self) -> None:
        """Reject calls to functions that were declared but never implemented."""
        unresolved = [
            symbol.name
            for symbol in self.function_map.values()
            if symbol.referenced and symbol.implementation is None
        ]
        if not unresolved:
            return

        unresolved.sort()
        if len(unresolved) == 1:
            raise CompileError(
                f"Function '{unresolved[0]}' is declared but has no implementation\n"
                "  A called user function must have a function definition"
            )

        raise CompileError(
            "Called user functions are declared but have no implementations\n"
            f"  Missing definitions: {', '.join(unresolved)}"
        )

    def validate_function_semantics(self) -> None:
        """Validate local scopes, names, types, and control flow without NCS emission."""
        from pykotor.resource.formats.ncs.compiler.semantic import (  # noqa: PLC0415
            FunctionSemanticAnalyzer,
        )

        for symbol in self.function_map.values():
            if symbol.implementation is not None:
                FunctionSemanticAnalyzer(self, symbol.implementation).validate()

    def compile(self, ncs: NCS):  # noqa: A003
        # BioWare expands includes before global/function processing. IncludeContext
        # provides include-once, active-stack recursion checks, and depth limiting.
        self.expand_includes()

        # Includes have now been expanded into this root. Build the complete
        # semantic symbol table and validate every function body before *any* NCS
        # bytecode is emitted. This keeps source diagnostics independent of dead-code
        # elimination and stack-layout details in the emitter.
        self.register_symbols()
        self.validate_function_semantics()

        script_globals: list[
            GlobalVariableDeclaration | GlobalVariableInitialization | StructDefinition
        ] = [
            obj
            for obj in self.objects
            if isinstance(
                obj,
                (GlobalVariableDeclaration, GlobalVariableInitialization, StructDefinition),
            )
        ]
        others: list[TopLevelObject] = [
            obj for obj in self.objects if obj not in script_globals
        ]

        if script_globals:
            for global_def in script_globals:
                global_def.compile(ncs, self)
            if self.scope_size() != 0:
                ncs.add(NCSInstructionType.SAVEBP, args=[])
        entry_index: int = len(ncs.instructions)

        for obj in others:
            obj.compile(ncs, self)

        # Prototypes are semantic declarations, not executable stubs.  Calls to a
        # prototype-only symbol are diagnosed only if the symbol was actually used.
        self.validate_referenced_functions()

        main_symbol = self.function_map.get("main")
        conditional_symbol = self.function_map.get("StartingConditional")

        if main_symbol is not None and main_symbol.has_root_declaration:
            self._validate_entry_point(main_symbol)
            entry_instruction = self._require_entry_implementation(main_symbol)
            ncs.add(NCSInstructionType.RETN, args=[], index=entry_index)
            ncs.add(
                NCSInstructionType.JSR,
                jump=entry_instruction,
                index=entry_index,
            )
        elif (
            conditional_symbol is not None
            and conditional_symbol.has_root_declaration
        ):
            self._validate_entry_point(conditional_symbol)
            entry_instruction = self._require_entry_implementation(conditional_symbol)
            ncs.add(NCSInstructionType.RETN, args=[], index=entry_index)
            ncs.add(
                NCSInstructionType.JSR,
                jump=entry_instruction,
                index=entry_index,
            )
            ncs.add(NCSInstructionType.RSADDI, args=[], index=entry_index)
        else:
            msg = (
                "This file has no entry point and cannot be compiled (Most likely an include file)."
            )
            raise EntryPointError(msg)

    @staticmethod
    def _validate_entry_point(symbol: FunctionSymbol) -> None:
        name = symbol.name
        signature = symbol.signature
        if name == "main":
            if signature.return_type != DynamicDataType.VOID:
                raise EntryPointError("Function 'main' must return void")
            if signature.parameter_types:
                raise EntryPointError("Function 'main' must have no parameters")
            return

        if name == "StartingConditional":
            if signature.return_type != DynamicDataType.INT:
                raise EntryPointError("Function 'StartingConditional' must return int")
            if signature.parameter_types:
                raise EntryPointError("Function 'StartingConditional' must have no parameters")
            return

        raise EntryPointError(f"Unsupported entry point '{name}'")

    @staticmethod
    def _require_entry_implementation(symbol: FunctionSymbol) -> NCSInstruction:
        """Return an entry label, reporting prototype-only entry points cleanly."""
        if symbol.implementation is None or symbol.entry_instruction is None:
            raise EntryPointError(
                f"Function '{symbol.name}' is declared as the script entry point "
                "but has no implementation"
            )
        return symbol.entry_instruction

    def get_engine_function(self, name: str) -> EngineFunctionReference | None:
        """Return predefined engine routine metadata for a source-level call name."""
        return self._engine_function_map.get(name)

    def get_visible_function(
        self,
        name: str,
        source_origin: SourceOrigin | None,
    ) -> FunctionSymbol | None:
        """Return a user function only if it was declared at this source position."""
        symbol = self.function_map.get(name)
        if symbol is None or not symbol.is_visible_at(source_origin):
            return None
        return symbol

    def callable_names(
        self,
        source_origin: SourceOrigin | None = None,
    ) -> tuple[str, ...]:
        """Return callable names visible at a source position for diagnostics."""
        user_names = tuple(
            name
            for name, symbol in self.function_map.items()
            if symbol.is_visible_at(source_origin)
        )
        return (*user_names, *self._engine_function_map)

    def compile_jsr(
        self,
        ncs: NCS,
        block: CodeBlock,
        symbol: FunctionSymbol,
        *args: Expression,
    ) -> DynamicDataType:
        args_list = list(args)

        name = symbol.name
        parameters = symbol.parameters
        return_type = symbol.signature.return_type
        start_instruction = symbol.mark_referenced()

        if len(args_list) > len(parameters):
            raise CompileError(
                f"Too many arguments in call to '{name}'\n"
                f"  Expected at most: {len(parameters)}\n"
                f"  Got: {len(args_list)}"
            )

        required_params = [param for param in parameters if param.default is None]

        # Make sure the minimal number of arguments were passed through
        if len(required_params) > len(args_list):
            required_names = [p.identifier.label for p in required_params]
            msg = (
                f"Missing required parameters in call to '{name}'\n"
                f"  Required: {', '.join(required_names)}\n"
                f"  Provided {len(args_list)} of {len(parameters)} parameters"
            )
            raise CompileError(msg)

        # If some optional parameters were not specified, add the defaults to the arguments list
        while len(parameters) > len(args_list):
            param_index = len(args_list)
            default_expr = parameters[param_index].default
            if default_expr is None:
                # Should not happen as required_params already checked, but be safe
                msg = f"Missing default value for parameter {param_index} in '{name}'"
                raise CompileError(msg)
            args_list.append(default_expr)

        # Reserve stack space for the return value and track it in the shared stack context
        return_type_size = 0
        if return_type == DynamicDataType.INT:
            ncs.add(NCSInstructionType.RSADDI, args=[])
            return_type_size = 4
        elif return_type == DynamicDataType.FLOAT:
            ncs.add(NCSInstructionType.RSADDF, args=[])
            return_type_size = 4
        elif return_type == DynamicDataType.STRING:
            ncs.add(NCSInstructionType.RSADDS, args=[])
            return_type_size = 4
        elif return_type == DynamicDataType.VECTOR:
            # Vectors are 3 floats (x, y, z components)
            # Reserve stack space for all 3 components
            ncs.add(NCSInstructionType.RSADDF, args=[])
            ncs.add(NCSInstructionType.RSADDF, args=[])
            ncs.add(NCSInstructionType.RSADDF, args=[])
            return_type_size = 12
        elif return_type == DynamicDataType.OBJECT:
            ncs.add(NCSInstructionType.RSADDO, args=[])
            return_type_size = 4
        elif return_type == DynamicDataType.TALENT:
            ncs.add(NCSInstructionType.RSADDTAL, args=[])
            return_type_size = 4
        elif return_type == DynamicDataType.EVENT:
            ncs.add(NCSInstructionType.RSADDEVT, args=[])
            return_type_size = 4
        elif return_type == DynamicDataType.LOCATION:
            ncs.add(NCSInstructionType.RSADDLOC, args=[])
            return_type_size = 4
        elif return_type == DynamicDataType.EFFECT:
            ncs.add(NCSInstructionType.RSADDEFF, args=[])
            return_type_size = 4
        elif return_type == DynamicDataType.VOID:
            return_type_size = 0
        elif return_type.builtin == DataType.STRUCT:
            # For struct return types, initialize the struct on the stack
            struct_name = return_type._struct  # noqa: SLF001
            if struct_name is not None and struct_name in self.struct_map:
                self.struct_map[struct_name].initialize(ncs, self)
                return_type_size = return_type.size(self)
            else:
                msg = "Unknown struct type for return value"
                raise CompileError(msg)
        else:
            msg = f"Trying to return unsupported type '{return_type.builtin.name}'"
            raise CompileError(msg)

        # The return slot is physically present before the arguments. Track it so
        # every argument sees correct SP-relative offsets.
        block.context.stack.push(return_type_size)

        offset = 0
        for param, arg in zip(parameters, args_list):
            arg_datatype: DynamicDataType = arg.compile(ncs, self, block)
            offset += arg_datatype.size(self)
            if param.data_type != arg_datatype:
                msg = (
                    f"Parameter type mismatch in call to '{name}'\n"
                    f"  Parameter '{param.identifier}' expects: {param.data_type.builtin.name}\n"
                    f"  Got: {arg_datatype.builtin.name}"
                )
                raise CompileError(msg)
        # JSR consumes the arguments but leaves the caller-allocated return slot.
        block.context.stack.consume(offset)
        ncs.add(NCSInstructionType.JSR, jump=start_instruction)

        return return_type

    def get_compile_time_constant(
        self,
        identifier: Identifier | str,
        source_origin: SourceOrigin | None = None,
    ) -> ConstantValue | None:
        label = identifier.label if isinstance(identifier, Identifier) else identifier
        symbol = self._compile_time_constants.get(label)
        if symbol is None or not symbol.is_visible_at(source_origin):
            return None
        return symbol.value

    def get_registered_global(
        self,
        identifier: Identifier | str,
        source_origin: SourceOrigin | None = None,
    ) -> ScopedValue | None:
        """Return source-level global metadata without requiring VM storage emission."""
        label = identifier.label if isinstance(identifier, Identifier) else identifier
        symbol = self._registered_globals.get(label)
        if (
            symbol is not None
            and symbol.is_const
            and self.get_compile_time_constant(label, source_origin) is None
        ):
            return None
        return symbol

    def registered_global_names(
        self,
        source_origin: SourceOrigin | None = None,
    ) -> tuple[str, ...]:
        """Return registered global names for source-level diagnostics."""
        return tuple(
            name
            for name, symbol in self._registered_globals.items()
            if not symbol.is_const
            or self.get_compile_time_constant(name, source_origin) is not None
        )

    def _define_compile_time_constant(
        self,
        identifier: Identifier,
        datatype: DynamicDataType,
        expression: Expression | None,
        origin: SourceOrigin,
    ) -> None:
        if datatype.builtin not in (DataType.INT, DataType.FLOAT, DataType.STRING):
            raise CompileError(
                f"Invalid type for const '{identifier}': {datatype.builtin.name.lower()}"
                "\n  BioWare-style const declarations support only int, float, and string"
            )

        if expression is None:
            defaults: dict[DataType, int | float | str] = {
                DataType.INT: 0,
                DataType.FLOAT: 0.0,
                DataType.STRING: "",
            }
            constant = ConstantValue(datatype.builtin, defaults[datatype.builtin])
        else:
            constant = expression.constant_value(self, origin)
            if constant is None:
                raise CompileError(
                    f"Invalid value assigned to constant '{identifier}'"
                    "\n  Const initializers must be compile-time constant expressions"
                )
            if constant.datatype != datatype.builtin:
                raise CompileError(
                    f"Type mismatch in constant '{identifier}'\n"
                    f"  Declared type: {datatype.builtin.name.lower()}\n"
                    f"  Initializer type: {constant.datatype.name.lower()}"
                )

        if constant.datatype == DataType.INT:
            constant = ConstantValue(DataType.INT, _int32(int(constant.value)))
        elif constant.datatype == DataType.FLOAT:
            constant = ConstantValue(DataType.FLOAT, _float32(float(constant.value)))
        else:
            constant = ConstantValue(DataType.STRING, str(constant.value))
        self._compile_time_constants[identifier.label] = CompileTimeConstantSymbol(
            constant,
            origin,
        )

    def allocate_global(
        self,
        identifier: Identifier,
        datatype: DynamicDataType,
        is_const: bool = False,
    ) -> None:
        """Record one VM global slot after its allocation instructions are emitted."""
        registered = self._registered_globals.get(identifier.label)
        if registered is None:
            raise ValueError(
                f"Internal compiler error: global '{identifier}' was emitted before registration"
            )
        if registered.data_type != datatype or registered.is_const != is_const:
            raise ValueError(
                f"Internal compiler error: emitted global '{identifier}' does not match registration"
            )
        if any(scoped.identifier == identifier for scoped in self._global_scope):
            raise ValueError(
                f"Internal compiler error: global '{identifier}' storage was emitted twice"
            )
        self._global_scope.insert(0, ScopedValue(identifier, datatype, is_const))

    def get_scoped(self, identifier: Identifier, root: CodeRoot) -> GetScopedResult:
        offset = 0
        for scoped in self._global_scope:
            offset -= scoped.data_type.size(root)
            if scoped.identifier == identifier:
                break
        else:
            # Provide helpful error with available globals
            available = [s.identifier.label for s in self._global_scope[:10]]  # Show first 10
            more = len(self._global_scope) - 10
            more_text = f" (and {more} more)" if more > 0 else ""
            msg = f"Undefined variable '{identifier}'\n  Available globals: {', '.join(available)}{more_text}"
            raise CompileError(msg)
        return GetScopedResult(
            is_global=True, datatype=scoped.data_type, offset=offset, is_const=scoped.is_const
        )

    def scope_size(self):
        return 0 - sum(scoped.data_type.size(self) for scoped in self._global_scope)


class CodeBlock:
    def __init__(self, context: CompilationContext | None = None):
        self.scope: list[ScopedValue] = []
        self._parent: CodeBlock | None = None
        self._statements: list[Statement] = []
        self._context: CompilationContext | None = context

    @property
    def context(self) -> CompilationContext:
        if self._context is None:
            self._context = CompilationContext()
        return self._context

    def add(self, statement: Statement):
        self._statements.append(statement)

    @property
    def statements(self) -> tuple[Statement, ...]:
        """Source statements in this block, exposed read-only to semantic analysis."""
        return tuple(self._statements)

    def compile(  # noqa: A003
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock | None,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
        *,
        context: CompilationContext | None = None,
    ):
        self._parent = block
        if block is not None:
            self._context = block.context
        elif context is not None:
            self._context = context
        elif self._context is None:
            self._context = CompilationContext()

        entry_transient_bytes = self.context.stack.snapshot()

        for statement in self._statements:
            statement.compile(
                ncs,
                root,
                self,
                return_instruction,
                break_instruction,
                continue_instruction,
            )
            if isinstance(statement, ReturnStatement):
                # Directly unreachable statements are omitted, matching the existing
                # dead-code behavior. ReturnStatement owns all return lowering so the
                # same logic also works when a return appears inside a switch body.
                self.context.stack.restore(entry_transient_bytes)
                return

        # External compiler optimizes away MOVSP with offset 0, so match it.
        scope_size = self.scope_size(root)
        if scope_size != 0:
            ncs.instructions.append(
                NCSInstruction(NCSInstructionType.MOVSP, [-scope_size]),
            )

        if self.context.stack.temporary_bytes != entry_transient_bytes:
            msg = (
                "Internal compiler error: Temporary stack changed across block compilation\n"
                f"  Entry temporary stack size: {entry_transient_bytes}\n"
                f"  Exit temporary stack size: {self.context.stack.temporary_bytes}\n"
                "  Every statement must leave the transient stack at its entry depth"
            )
            raise ValueError(msg)

    def add_scoped(
        self, identifier: Identifier, data_type: DynamicDataType, is_const: bool = False
    ):
        self.scope.insert(0, ScopedValue(identifier, data_type, is_const))

    def get_scoped(
        self,
        identifier: Identifier,
        root: CodeRoot,
        offset: int | None = None,
    ) -> GetScopedResult:
        # The shared transient-stack depth is subtracted exactly once at the
        # innermost lookup. Parent-scope recursion only walks persistent locals.
        offset = -self.context.stack.temporary_bytes if offset is None else offset
        for scoped in self.scope:
            offset -= scoped.data_type.size(root)
            if scoped.identifier == identifier:
                break
        else:
            if self._parent is not None:
                return self._parent.get_scoped(identifier, root, offset)
            return root.get_scoped(identifier, root)
        return GetScopedResult(
            is_global=False, datatype=scoped.data_type, offset=offset, is_const=scoped.is_const
        )

    def scope_size(self, root: CodeRoot) -> int:
        """Return the byte size of values owned by this lexical scope."""
        return sum(scoped.data_type.size(root) for scoped in self.scope)

    def full_scope_size(self, root: CodeRoot) -> int:
        """Return local/parameter bytes from this block through the function root."""
        size = self.scope_size(root)
        if self._parent is not None:
            size += self._parent.full_scope_size(root)
        return size

    def stack_depth(self, root: CodeRoot) -> int:
        """Return the tracked VM depth relevant to lexical control-flow unwinding."""
        return self.full_scope_size(root) + self.context.stack.temporary_bytes

    def returns_on_all_paths(self) -> bool:
        """Match BioWare's conservative non-void return-path analysis.

        The original compiler only proves total return coverage through direct
        returns, statement/compound-statement lists, scoped blocks, and complete
        if/else choices.  It deliberately does *not* try to prove that loops or
        switches always return, even when that may be obvious to a human reader.
        """
        return any(_statement_returns_on_all_paths(statement) for statement in self._statements)

    def unwind_size_to(self, root: CodeRoot, target_depth: int) -> int:
        current_depth = self.stack_depth(root)
        if target_depth > current_depth:
            raise ValueError(
                "Internal compiler error: control-flow target is deeper than current stack "
                f"({target_depth} > {current_depth})"
            )
        return current_depth - target_depth


class ScopedValue:
    def __init__(self, identifier: Identifier, data_type: DynamicDataType, is_const: bool = False):
        self.identifier: Identifier = identifier
        self.data_type: DynamicDataType = data_type
        self.is_const: bool = is_const


class FunctionForwardDeclaration(TopLevelObject):
    def __init__(
        self,
        return_type: DynamicDataType,
        identifier: Identifier,
        parameters: list[FunctionDefinitionParam],
    ):
        self.return_type: DynamicDataType = return_type
        self.identifier: Identifier = identifier
        self.parameters: list[FunctionDefinitionParam] = parameters

    def register(self, root: CodeRoot) -> None:
        root.register_function(self)

    def compile(self, ncs: NCS, root: CodeRoot):  # noqa: A003
        # A prototype is semantic information only.  It deliberately emits no NCS
        # instruction and therefore cannot become an accidental JSR destination.
        return


class FunctionDefinition(TopLevelObject):
    """Represents a function definition with implementation.

    Contains the function signature (return type, parameters) and the code block
    that implements the function body.

    Signature matching is handled by :class:`FunctionSignature` during symbol
    registration; this node owns only the implementation's source body and metadata.
    """

    def __init__(
        self,
        return_type: DynamicDataType,
        identifier: Identifier,
        parameters: list[FunctionDefinitionParam],
        block: CodeBlock,
        line_num: int,
    ):
        self.return_type: DynamicDataType = return_type
        self.identifier: Identifier = identifier
        self.parameters: list[FunctionDefinitionParam] = parameters
        self.block: CodeBlock = block
        self.line_num: int = line_num

        for param in parameters:
            block.add_scoped(param.identifier, param.data_type)

    def register(self, root: CodeRoot) -> None:
        root.register_function(self)

    def compile(self, ncs: NCS, root: CodeRoot):  # noqa: A003
        name = self.identifier.label
        symbol = root.function_map[name]
        if symbol.implementation is not self:
            raise ValueError(
                f"Internal compiler error: function '{name}' does not match its registered implementation"
            )
        retn = NCSInstruction(NCSInstructionType.RETN)
        ncs.instructions.append(symbol.require_entry_instruction())
        context = CompilationContext(
            SemanticContext(name, self.return_type, self.require_source_origin())
        )
        self.block.compile(ncs, root, None, retn, None, None, context=context)
        ncs.instructions.append(retn)


class FunctionDefinitionParam:
    def __init__(
        self,
        data_type: DynamicDataType,
        identifier: Identifier,
        default: Expression | None = None,
    ):
        self.data_type: DynamicDataType = data_type
        self.identifier: Identifier = identifier
        self.default: Expression | None = default


def _fold_default_parameter_expression(
    parameter: FunctionDefinitionParam,
    root: CodeRoot,
    function_name: str,
    source_origin: SourceOrigin,
) -> None:
    """Validate and canonicalize one BioWare-style optional parameter value."""
    expression = parameter.default
    if expression is None:
        return

    datatype = parameter.data_type.builtin
    parameter_name = parameter.identifier.label

    if datatype in (DataType.INT, DataType.FLOAT, DataType.STRING):
        constant = expression.constant_value(root, source_origin)
        if constant is None:
            raise CompileError(
                f"Non-constant default value for parameter '{parameter_name}' in '{function_name}'"
                "\n  Default arguments must be compile-time constant expressions"
            )
        if constant.datatype != datatype:
            raise CompileError(
                f"Default value type mismatch for parameter '{parameter_name}' in '{function_name}'"
                f"\n  Expected: {datatype.name.lower()}"
                f"\n  Got: {constant.datatype.name.lower()}"
            )

        if datatype == DataType.INT:
            parameter.default = IntExpression(_int32(int(constant.value)))
        elif datatype == DataType.FLOAT:
            parameter.default = FloatExpression(_float32(float(constant.value)))
        else:
            parameter.default = StringExpression(str(constant.value))
        return

    # BioWare permits object and vector defaults only as literal constants. They do
    # not participate in the int/float/string compile-time constant symbol table.
    if datatype == DataType.OBJECT:
        if not isinstance(expression, ObjectExpression):
            raise CompileError(
                f"Non-constant default value for parameter '{parameter_name}' in '{function_name}'"
                "\n  Object defaults must be OBJECT_SELF or OBJECT_INVALID"
            )
        parameter.default = ObjectExpression(expression.value)
        return

    if datatype == DataType.VECTOR:
        if not isinstance(expression, VectorExpression):
            raise CompileError(
                f"Non-constant default value for parameter '{parameter_name}' in '{function_name}'"
                "\n  Vector defaults must be vector literals"
            )
        parameter.default = VectorExpression(
            FloatExpression(_float32(expression.x.value)),
            FloatExpression(_float32(expression.y.value)),
            FloatExpression(_float32(expression.z.value)),
        )
        return

    raise CompileError(
        f"Type '{datatype.name.lower()}' does not support default parameters"
        f"\n  Parameter: {parameter_name} in '{function_name}'"
    )


def _validate_and_fold_default_parameters(
    parameters: list[FunctionDefinitionParam],
    root: CodeRoot,
    function_name: str,
    source_origin: SourceOrigin,
) -> None:
    """Apply BioWare optional-parameter ordering, constness, and type rules."""
    optional_parameters_started = False
    for parameter in parameters:
        if parameter.default is None:
            if optional_parameters_started:
                raise CompileError(
                    "Function parameter without a default value can't follow one with a default value."
                )
            continue

        optional_parameters_started = True
        _fold_default_parameter_expression(
            parameter,
            root,
            function_name,
            source_origin,
        )


class IncludeScript(TopLevelObject):
    def __init__(
        self,
        file: StringExpression,
        library: dict[str, bytes] | None = None,
    ):
        self.file: StringExpression = file
        self.library: dict[str, bytes] = {} if library is None else library

    def expand(
        self,
        root: CodeRoot,
        context: IncludeContext,
    ) -> list[TopLevelObject]:
        """Parse this include and recursively expand its nested includes."""
        canonical_name = context.begin_include(self.file.value)
        if canonical_name is None:
            return []

        completed = False
        try:
            included_root = self._parse(root)
            included_file_level = len(context.active) + 1
            expanded = root._expand_include_objects(
                included_root.objects,
                file_level=included_file_level,
            )
            completed = True
            return expanded
        finally:
            context.end_include(canonical_name, completed=completed)

    def _parse(self, root: CodeRoot) -> CodeRoot:
        from pykotor.resource.formats.ncs.compiler.lexer import NssLexer  # noqa: PLC0415
        from pykotor.resource.formats.ncs.compiler.parser import NssParser  # noqa: PLC0415

        lookup_paths = cast(
            "list[str] | None",
            [str(path) for path in root.library_lookup] if root.library_lookup else None,
        )
        parser = NssParser(
            root.functions,
            root.constants,
            root.library,
            lookup_paths,
            max_include_depth=root.include_context.max_depth,
        )
        source = self._get_script(root)
        lexer = NssLexer()
        return parser.parser.parse(source, lexer=lexer.lexer, tracking=True)

    def compile(self, ncs: NCS, root: CodeRoot):  # noqa: A003
        raise RuntimeError(
            "Internal compiler error: include directive reached bytecode emission"
        )

    def _get_script(self, root: CodeRoot) -> str:
        """Load an included script using KOTOR's case-insensitive resource names."""
        for folder in root.library_lookup:
            filepath = folder / f"{self.file.value}.nss"
            if filepath.is_file():
                try:
                    return filepath.read_bytes().decode(errors="ignore")
                except Exception as exc:
                    raise MissingIncludeError(
                        f"Failed to read include file '{filepath}': {exc}"
                    ) from exc

        canonical_name = IncludeContext.canonicalize(self.file.value)
        for library_name, source_bytes in self.library.items():
            if IncludeContext.canonicalize(library_name) == canonical_name:
                return source_bytes.decode(errors="ignore")

        search_paths = [str(folder) for folder in root.library_lookup]
        raise MissingIncludeError(
            f"Could not find included script '{self.file.value}.nss'\n"
            f"  Searched in {len(search_paths)} path(s): {', '.join(search_paths[:3])}"
            f"{'...' if len(search_paths) > 3 else ''}\n"
            f"  Also checked {len(self.library)} library file(s)"
        )


class StructDefinition(TopLevelObject):
    def __init__(self, identifier: Identifier, members: list[StructMember]):
        self.identifier: Identifier = identifier
        self.members: list[StructMember] = members

    def register(self, root: CodeRoot) -> None:
        root.register_struct(self)

    def compile(self, ncs: NCS, root: CodeRoot):  # noqa: A003
        # Structs are compile-time-only symbols. Their complete validation/layout
        # is performed during registration and they emit no NCS instructions.
        return None


class Expression(ABC):
    """Abstract base class for NSS expressions.

    Expressions compile to NCS bytecode instructions that evaluate to values.
    All expression types (literals, operators, function calls, etc.) inherit from this.

    References:
    ----------

    """

    def constant_value(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None = None,
    ) -> ConstantValue | None:
        """Return the compile-time value of this expression, if one exists."""
        return None

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
    ) -> DynamicDataType:
        """Compile an expression and enforce one stack-accounting contract.

        Every expression leaves exactly one value of its reported type on the VM
        stack (or zero bytes for ``void``). Child expressions may update the shared
        context while they are being lowered, but callers never need to guess
        whether a particular node tracked its own result.
        """
        entry_stack = block.context.stack.snapshot()
        data_type = self._compile(ncs, root, block)
        block.context.stack.restore(entry_stack + data_type.size(root))
        return data_type

    @abstractmethod
    def _compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
    ) -> DynamicDataType: ...


class Statement(ABC):
    """Abstract base class for NSS statements.

    Statements compile to NCS bytecode instructions that perform actions (control flow,
    assignments, declarations, etc.). All statement types inherit from this.

    References:
    ----------

    """

    def __init__(self):
        self.line_num: None = None

    @abstractmethod
    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ) -> object: ...


class FieldAccess:
    def __init__(self, identifiers: list[Identifier]):
        super().__init__()
        self.identifiers: list[Identifier] = identifiers

    def get_scoped(self, block: CodeBlock, root: CodeRoot) -> GetScopedResult:
        """Get scoped variable information for field access.

        Args:
        ----
            block: Current code block
            root: Code root context

        Returns:
        -------
            GetScopedResult: Variable scope information

        Raises:
        ------
            CompileError: If field access is invalid
        """
        if len(self.identifiers) == 0:
            msg = "Internal error: FieldAccess has no identifiers"
            raise CompileError(msg)

        first_ident: Identifier = self.identifiers[0]
        if root.get_compile_time_constant(
            first_ident,
            block.context.semantic.source_origin,
        ) is not None:
            raise CompileError(f"Cannot assign to compile-time constant '{first_ident}'")
        scoped: GetScopedResult = block.get_scoped(first_ident, root)

        is_global: bool = scoped.is_global
        offset: int = scoped.offset
        datatype: DynamicDataType = scoped.datatype
        is_const: bool = scoped.is_const  # Get is_const from the first call

        for next_ident in self.identifiers[1:]:
            # Check previous datatype to see what members are accessible
            if datatype.builtin == DataType.VECTOR:
                datatype = DynamicDataType.FLOAT
                if next_ident.label == "x":
                    offset += 0
                elif next_ident.label == "y":
                    offset += 4
                elif next_ident.label == "z":
                    offset += 8
                else:
                    msg = f"Attempting to access unknown member '{next_ident}' on datatype '{datatype}'."
                    raise CompileError(msg)
            elif datatype.builtin == DataType.STRUCT:
                assert datatype._struct is not None, (
                    "datatype._struct cannot be None in FieldAccess.get_scoped()"
                )  # noqa: SLF001
                offset += root.struct_map[datatype._struct].child_offset(  # noqa: SLF001
                    root,
                    next_ident,
                )
                datatype = root.struct_map[datatype._struct].child_type(  # noqa: SLF001
                    root,
                    next_ident,
                )
            else:
                msg = (
                    f"Attempting to access unknown member '{next_ident}' on datatype '{datatype}'."
                )
                raise CompileError(msg)

        return GetScopedResult(is_global, datatype, offset, is_const)

    def compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        is_global, variable_type, stack_index, _is_const = self.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))
        return variable_type


# region Expressions: Simple
class IdentifierExpression(Expression):
    def __init__(self, value: Identifier):
        super().__init__()
        self.identifier: Identifier = value

    def __eq__(self, other: IdentifierExpression | object) -> bool:
        if self is other:
            return True
        if isinstance(other, IdentifierExpression):
            return self.identifier == other.identifier
        return NotImplemented  # type: ignore[no-any-return]

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __repr__(self) -> str:
        return f"IdentifierExpression(identifier={self.identifier})"

    def constant_value(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None = None,
    ) -> ConstantValue | None:
        return root.get_compile_time_constant(self.identifier, source_origin)

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        constant = self.constant_value(root, block.context.semantic.source_origin)
        if constant is not None:
            return _emit_constant(ncs, constant)

        is_global, datatype, stack_index, _is_const = block.get_scoped(self.identifier, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, datatype.size(root)])
        return datatype

    def get_constant(self, root: CodeRoot) -> ScriptConstant | None:
        return next(
            (constant for constant in root.constants if constant.name == self.identifier.label),
            None,
        )

    def is_constant(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None = None,
    ) -> bool:
        return self.constant_value(root, source_origin) is not None


class FieldAccessExpression(Expression):
    def __init__(self, field_access: FieldAccess):
        super().__init__()
        self.field_access: FieldAccess = field_access

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        scoped = self.field_access.get_scoped(block, root)
        instruction_type = (
            NCSInstructionType.CPTOPBP if scoped.is_global else NCSInstructionType.CPTOPSP
        )
        ncs.instructions.append(
            NCSInstruction(
                instruction_type,
                [scoped.offset, scoped.datatype.size(root)],
            ),
        )
        return scoped.datatype


class StringExpression(Expression):
    def __init__(self, value: str):
        super().__init__()
        self.value: str = value

    def __eq__(self, other: StringExpression | object):
        if self is other:
            return True
        if isinstance(other, StringExpression):
            return self.value == other.value
        return NotImplemented  # type: ignore[no-any-return]

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"StringExpression(value={self.value})"

    def data_type(self) -> DynamicDataType:
        return DynamicDataType.STRING

    def constant_value(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None = None,
    ) -> ConstantValue | None:
        return ConstantValue(DataType.STRING, self.value)

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        ncs.instructions.append(NCSInstruction(NCSInstructionType.CONSTS, [self.value]))
        return DynamicDataType.STRING


class IntExpression(Expression):
    def __init__(self, value: int):
        super().__init__()
        self.value: int = value

    def __eq__(self, other: IntExpression | object):
        if self is other:
            return True
        if isinstance(other, IntExpression):
            return self.value == other.value
        return NotImplemented  # type: ignore[no-any-return]

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"IntExpression(value={self.value})"

    def data_type(self) -> DynamicDataType:
        return DynamicDataType.INT

    def constant_value(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None = None,
    ) -> ConstantValue | None:
        return ConstantValue(DataType.INT, self.value)

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        # NCS CONSTI is a signed 32-bit field. Newer compiler sources accept the full
        # 32-bit hexadecimal range (for example 0x80000000 for unsigned shifts),
        # while this branch's binary writer expects an already-signed Python int.
        value = self.value & 0xFFFFFFFF
        if value >= 0x80000000:
            value -= 0x100000000
        ncs.instructions.append(NCSInstruction(NCSInstructionType.CONSTI, [value]))
        return DynamicDataType.INT


class ObjectExpression(Expression):
    def __init__(self, value: int):
        super().__init__()
        self.value: int = value

    def __eq__(self, other: ObjectExpression | object):
        if self is other:
            return True
        if isinstance(other, ObjectExpression):
            return self.value == other.value
        return NotImplemented  # type: ignore[no-any-return]

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"ObjectExpression(value={self.value})"

    def data_type(self) -> DynamicDataType:
        return DynamicDataType.OBJECT

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        ncs.instructions.append(NCSInstruction(NCSInstructionType.CONSTO, [self.value]))
        return DynamicDataType.OBJECT


class FloatExpression(Expression):
    def __init__(self, value: float):
        super().__init__()
        self.value: float = value

    def __eq__(self, other: FloatExpression | object):
        if self is other:
            return True
        if isinstance(other, FloatExpression):
            return self.value == other.value
        return NotImplemented  # type: ignore[no-any-return]

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"FloatExpression(value={self.value})"

    def data_type(self) -> DynamicDataType:
        return DynamicDataType.FLOAT

    def constant_value(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None = None,
    ) -> ConstantValue | None:
        return ConstantValue(DataType.FLOAT, self.value)

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        ncs.instructions.append(NCSInstruction(NCSInstructionType.CONSTF, [self.value]))
        return DynamicDataType.FLOAT


class VectorExpression(Expression):
    def __init__(self, x: FloatExpression, y: FloatExpression, z: FloatExpression):
        super().__init__()
        self.x: FloatExpression = x
        self.y: FloatExpression = y
        self.z: FloatExpression = z

    def __eq__(self, other: VectorExpression | object):
        if self is other:
            return True
        if isinstance(other, VectorExpression):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return NotImplemented  # type: ignore[no-any-return]

    def __hash__(self) -> int:
        return hash(self.x) ^ hash(self.y) ^ hash(self.z)

    def __repr__(self) -> str:
        return f"VectorExpression(x={self.x}, y={self.y}, z={self.z})"

    def data_type(self) -> DynamicDataType:
        return DynamicDataType.VECTOR

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        self.x.compile(ncs, root, block)
        self.y.compile(ncs, root, block)
        self.z.compile(ncs, root, block)
        return DynamicDataType.VECTOR


class EngineCallExpression(Expression):
    """Explicit engine-call node retained for callers that construct ASTs directly.

    Parsed NSS uses :class:`FunctionCallExpression` for every source call and resolves
    engine-versus-user ownership during semantic analysis/emission.
    """

    def __init__(
        self,
        function: ScriptFunction,
        routine_id: int,
        args: list[Expression],
    ):
        super().__init__()
        self._function: ScriptFunction = function
        self._routine_id: int = routine_id
        # Engine-call arguments are part of the parsed/constructed AST. Keep the
        # stored sequence immutable; default expansion happens on a local list in
        # ``_compile`` and must never change this node between compiler passes.
        self._args: tuple[Expression, ...] = tuple(args)

    @property
    def function(self) -> ScriptFunction:
        """Engine routine metadata used by semantic analysis and emission."""
        return self._function

    @property
    def arguments(self) -> tuple[Expression, ...]:
        """Explicit source arguments, excluding any materialized defaults."""
        return self._args

    def _compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
    ) -> DynamicDataType:  # noqa: A003
        # Default materialization is an emission detail. Keep the parsed AST stable
        # across semantic analysis and any repeated compiler passes.
        args = list(self._args)
        arg_count = len(args)

        if arg_count > len(self._function.params):
            msg = (
                f"Too many arguments for '{self._function.name}'\n"
                f"  Expected: {len(self._function.params)}, Got: {arg_count}"
            )
            raise CompileError(msg)

        for i, param in enumerate(self._function.params):
            if i < arg_count:
                continue
            if param.default is None:
                required_params = [
                    p.name for p in self._function.params if p.default is None
                ]
                msg = (
                    f"Missing required arguments for '{self._function.name}'\n"
                    f"  Required parameters: {', '.join(required_params)}\n"
                    f"  Provided: {arg_count} argument(s)"
                )
                raise CompileError(msg)

            constant: ScriptConstant | None = next(
                (constant for constant in root.constants if constant.name == param.default),
                None,
            )
            if constant is None:
                if param.datatype == DataType.INT:
                    args.append(IntExpression(int(param.default)))
                elif param.datatype == DataType.FLOAT:
                    args.append(FloatExpression(float(param.default)))
                elif param.datatype == DataType.STRING:
                    args.append(StringExpression(param.default))
                elif param.datatype == DataType.VECTOR:
                    x = FloatExpression(param.default.x)
                    y = FloatExpression(param.default.y)
                    z = FloatExpression(param.default.z)
                    args.append(VectorExpression(x, y, z))
                elif param.datatype == DataType.OBJECT:
                    args.append(ObjectExpression(int(param.default)))
                else:
                    msg = (
                        f"Unsupported default parameter type '{param.datatype.name}' "
                        f"for '{param.name}' in '{self._function.name}'\n"
                        "  This may indicate a compiler limitation"
                    )
                    raise CompileError(msg)
            elif constant.datatype == DataType.INT:
                args.append(IntExpression(int(constant.value)))
            elif constant.datatype == DataType.FLOAT:
                args.append(FloatExpression(float(constant.value)))
            elif constant.datatype == DataType.STRING:
                args.append(StringExpression(str(constant.value)))
            elif constant.datatype == DataType.OBJECT:
                args.append(ObjectExpression(int(constant.value)))

        ordinary_argument_bytes = 0

        # BioWare ACTION calling convention places the first parameter at the
        # top of the VM stack. Emit arguments in reverse declaration order so
        # ACTION consumers pop parameter 0 first.
        for reverse_index, arg in enumerate(reversed(args)):
            param_index = len(args) - 1 - reverse_index
            param = self._function.params[param_index]
            param_type = DynamicDataType(param.datatype)
            if param_type == DataType.ACTION:
                after_command = NCSInstruction()
                ncs.add(
                    NCSInstructionType.STORE_STATE,
                    args=[-root.scope_size(), block.full_scope_size(root)],
                )
                ncs.add(NCSInstructionType.JMP, jump=after_command)
                action_entry_stack = block.context.stack.snapshot()
                actual_type = arg.compile(ncs, root, block)
                block.context.stack.restore(action_entry_stack)
                if actual_type != DynamicDataType.VOID:
                    raise CompileError(
                        f"ACTION parameter '{param.name}' in call to "
                        f"'{self._function.name}' must be a void expression\n"
                        f"  Got: {actual_type.builtin.name.lower()}"
                    )
                ncs.add(NCSInstructionType.RETN)
                ncs.instructions.append(after_command)
                continue

            actual_type = arg.compile(ncs, root, block)
            ordinary_argument_bytes += actual_type.size(root)
            if actual_type != param_type:
                raise CompileError(
                    f"Type mismatch for parameter '{param.name}' in call to "
                    f"'{self._function.name}'\n"
                    f"  Expected: {param_type.builtin.name.lower()}\n"
                    f"  Got: {actual_type.builtin.name.lower()}"
                )

        ncs.instructions.append(
            NCSInstruction(
                NCSInstructionType.ACTION,
                [self._routine_id, len(args)],
            ),
        )
        # ACTION consumes ordinary value arguments. Captured ACTION parameters are
        # stored as code/state and therefore do not contribute ordinary stack bytes.
        block.context.stack.consume(ordinary_argument_bytes)
        return DynamicDataType(self._function.returntype)


class FunctionCallExpression(Expression):
    def __init__(self, function: Identifier, args: list[Expression]):
        super().__init__()
        self._function: Identifier = function
        self._args: list[Expression] = args

    @property
    def function_identifier(self) -> Identifier:
        """Source identifier of the callable being invoked."""
        return self._function

    @property
    def arguments(self) -> tuple[Expression, ...]:
        """Explicit call arguments as a read-only semantic view."""
        return tuple(self._args)

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        name = self._function.label
        source_origin = block.context.semantic.source_origin
        symbol = root.get_visible_function(name, source_origin)
        if symbol is not None:
            # compile_jsr handles return-slot reservation and shared stack accounting.
            return root.compile_jsr(ncs, block, symbol, *self._args)

        engine = root.get_engine_function(name)
        if engine is not None:
            # Parsed calls stay unresolved until semantic analysis. Reuse the explicit
            # engine-call emitter without changing the source AST.
            engine_call = EngineCallExpression(
                engine.function,
                engine.routine_id,
                list(self._args),
            )
            return engine_call._compile(ncs, root, block)  # noqa: SLF001

        available = root.callable_names(source_origin)
        preview = available[:10]
        suffix = "..." if len(available) > 10 else ""
        raise CompileError(
            f"Undefined function '{name}'\n"
            f"  Available functions: {', '.join(preview)}{suffix}"
        )


# endregion


class BinaryOperatorExpression(Expression):
    def __init__(
        self,
        expression1: Expression,
        expression2: Expression,
        mapping: list[BinaryOperatorMapping],
    ):
        self.expression1: Expression = expression1
        self.expression2: Expression = expression2
        self.compatibility: list[BinaryOperatorMapping] = mapping

    def constant_value(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None = None,
    ) -> ConstantValue | None:
        left = self.expression1.constant_value(root, source_origin)
        if left is None:
            return None

        # Match BioWare's compile-time short-circuit behavior: the right side does
        # not need to be constant if the left side determines the logical result.
        if left.datatype == DataType.INT:
            for mapping in self.compatibility:
                if mapping.instruction == NCSInstructionType.LOGORII and int(left.value) != 0:
                    return ConstantValue(DataType.INT, 1)
                if mapping.instruction == NCSInstructionType.LOGANDII and int(left.value) == 0:
                    return ConstantValue(DataType.INT, 0)

        right = self.expression2.constant_value(root, source_origin)
        if right is None:
            return None
        for mapping in self.compatibility:
            folded = _fold_binary_constant(mapping, left, right)
            if folded is not None:
                return folded
        return None

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        constant = self.constant_value(root, block.context.semantic.source_origin)
        if constant is not None:
            return _emit_constant(ncs, constant)

        short_circuit_mapping = next(
            (
                mapping
                for mapping in self.compatibility
                if mapping.instruction
                in {NCSInstructionType.LOGANDII, NCSInstructionType.LOGORII}
            ),
            None,
        )
        if short_circuit_mapping is not None:
            return self._compile_short_circuit(
                ncs,
                root,
                block,
                short_circuit_mapping,
            )

        type1 = self.expression1.compile(ncs, root, block)
        type1_size = type1.size(root)
        type2 = self.expression2.compile(ncs, root, block)
        type2_size = type2.size(root)

        result_type: DynamicDataType | None = None
        for mapping in self.compatibility:
            if type1 == mapping.lhs and type2 == mapping.rhs:
                ncs.add(mapping.instruction)
                result_type = DynamicDataType(mapping.result)
                break

        if result_type is None:
            # User-defined structures (and BioWare's built-in vector structure)
            # use the generic STRUCT/STRUCT equality opcode with an explicit byte
            # count. Different named structures are intentionally incompatible.
            equality_instruction: NCSInstructionType | None = None
            if type1 == type2 and type1.builtin in {DataType.STRUCT, DataType.VECTOR}:
                instructions = {mapping.instruction for mapping in self.compatibility}
                if instructions & {
                    NCSInstructionType.EQUALII,
                    NCSInstructionType.EQUALFF,
                    NCSInstructionType.EQUALOO,
                    NCSInstructionType.EQUALSS,
                    NCSInstructionType.EQUALEFFEFF,
                    NCSInstructionType.EQUALEVTEVT,
                    NCSInstructionType.EQUALLOCLOC,
                    NCSInstructionType.EQUALTALTAL,
                }:
                    equality_instruction = NCSInstructionType.EQUALTT
                elif instructions & {
                    NCSInstructionType.NEQUALII,
                    NCSInstructionType.NEQUALFF,
                    NCSInstructionType.NEQUALOO,
                    NCSInstructionType.NEQUALSS,
                    NCSInstructionType.NEQUALEFFEFF,
                    NCSInstructionType.NEQUALEVTEVT,
                    NCSInstructionType.NEQUALLOCLOC,
                    NCSInstructionType.NEQUALTALTAL,
                }:
                    equality_instruction = NCSInstructionType.NEQUALTT

            if equality_instruction is not None:
                ncs.add(equality_instruction, args=[type1.size(root)])
                result_type = DynamicDataType.INT
            else:
                # Build helpful error showing what operations are supported.
                supported = [
                    f"{m.lhs.name.lower()} {m.instruction.name} {m.rhs.name.lower()}"
                    for m in self.compatibility[:3]
                ]
                msg = (
                    f"Incompatible types for binary operation: {type1.builtin.name.lower()} and {type2.builtin.name.lower()}\n"
                    f"  Supported combinations: {', '.join(supported)}"
                    f"{'...' if len(self.compatibility) > 3 else ''}"
                )
                raise CompileError(msg)

        result_size = result_type.size(root)
        # Binary instructions consume both operands and leave one result.
        block.context.stack.replace(type1_size + type2_size, result_size)
        return result_type

    def _compile_short_circuit(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        mapping: BinaryOperatorMapping,
    ) -> DynamicDataType:
        """Lower BioWare-style runtime ``&&``/``||`` short-circuiting.

        BioWare keeps the original left operand on the stack, duplicates it for
        the conditional jump, and only evaluates the right operand when needed.
        If the branch is short-circuited, the original left operand itself is the
        expression result; otherwise LOGANDII/LOGORII combines both operands.
        """
        type1 = self.expression1.compile(ncs, root, block)
        if type1 != mapping.lhs:
            raise CompileError(
                "Incompatible left operand for logical operation: "
                f"{type1.builtin.name.lower()}\n"
                f"  Expected: {mapping.lhs.name.lower()}"
            )

        lhs_size = type1.size(root)
        result_type = DynamicDataType(mapping.result)
        result_size = result_type.size(root)
        if lhs_size != result_size:
            raise ValueError(
                "Internal compiler error: short-circuit result size differs from left operand"
            )

        end_label = NCSInstruction(NCSInstructionType.NOP, args=[])

        # Duplicate only the test value. JZ/JNZ consumes the duplicate and leaves
        # the original lhs in place for either the short-circuit result or the
        # eventual LOGANDII/LOGORII operation.
        ncs.add(NCSInstructionType.CPTOPSP, args=[-lhs_size, lhs_size])
        block.context.stack.push(lhs_size)
        jump_type = (
            NCSInstructionType.JZ
            if mapping.instruction == NCSInstructionType.LOGANDII
            else NCSInstructionType.JNZ
        )
        ncs.add(jump_type, jump=end_label)
        block.context.stack.consume(lhs_size)

        type2 = self.expression2.compile(ncs, root, block)
        if type2 != mapping.rhs:
            raise CompileError(
                "Incompatible right operand for logical operation: "
                f"{type2.builtin.name.lower()}\n"
                f"  Expected: {mapping.rhs.name.lower()}"
            )

        rhs_size = type2.size(root)
        ncs.add(mapping.instruction)
        block.context.stack.replace(lhs_size + rhs_size, result_size)
        ncs.instructions.append(end_label)
        return result_type


class TernaryConditionalExpression(Expression):
    def __init__(self, condition: Expression, true_expr: Expression, false_expr: Expression):
        super().__init__()
        self.condition: Expression = condition
        self.true_expr: Expression = true_expr
        self.false_expr: Expression = false_expr

    def constant_value(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None = None,
    ) -> ConstantValue | None:
        condition = self.condition.constant_value(root, source_origin)
        if condition is None or condition.datatype != DataType.INT:
            return None
        true_value = self.true_expr.constant_value(root, source_origin)
        false_value = self.false_expr.constant_value(root, source_origin)
        if true_value is None or false_value is None or true_value.datatype != false_value.datatype:
            return None
        return true_value if int(condition.value) != 0 else false_value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        constant = self.constant_value(root, block.context.semantic.source_origin)
        if constant is not None:
            return _emit_constant(ncs, constant)

        initial_stack = block.context.stack.snapshot()

        condition_type = self.condition.compile(ncs, root, block)
        if condition_type != DynamicDataType.INT:
            msg = f"Ternary condition must be integer type, got {condition_type.builtin.name}\n  Note: Conditions must evaluate to int (0 = false, non-zero = true)"
            raise CompileError(msg)

        false_label = NCSInstruction(NCSInstructionType.NOP, args=[])
        ncs.add(NCSInstructionType.JZ, jump=false_label)
        block.context.stack.consume(condition_type.size(root))

        true_type = self.true_expr.compile(ncs, root, block)
        end_label = NCSInstruction(NCSInstructionType.NOP, args=[])
        ncs.add(NCSInstructionType.JMP, jump=end_label)

        ncs.instructions.append(false_label)
        block.context.stack.restore(initial_stack)
        false_type = self.false_expr.compile(ncs, root, block)

        if true_type != false_type:
            msg = (
                f"Type mismatch in ternary operator\n"
                f"  True branch type: {true_type.builtin.name}\n"
                f"  False branch type: {false_type.builtin.name}\n"
                f"  Both branches must have the same type"
            )
            raise CompileError(msg)

        ncs.instructions.append(end_label)
        block.context.stack.restore(initial_stack + true_type.size(root))
        return true_type


class UnaryOperatorExpression(Expression):
    def __init__(self, expression1: Expression, mapping: list[UnaryOperatorMapping]):
        super().__init__()
        self.expression1: Expression = expression1
        self.compatibility: list[UnaryOperatorMapping] = mapping

    def constant_value(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None = None,
    ) -> ConstantValue | None:
        operand = self.expression1.constant_value(root, source_origin)
        if operand is None:
            return None
        for mapping in self.compatibility:
            if operand.datatype == mapping.rhs:
                return _fold_unary_constant(mapping.instruction, operand)
        return None

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        constant = self.constant_value(root, block.context.semantic.source_origin)
        if constant is not None:
            return _emit_constant(ncs, constant)

        type1 = self.expression1.compile(ncs, root, block)

        for x in self.compatibility:
            if type1 == x.rhs:
                ncs.add(x.instruction)
                break
        else:
            supported_types = [m.rhs.name.lower() for m in self.compatibility]
            msg = f"Incompatible type for unary operation: {type1.builtin.name.lower()}\n  Supported types: {', '.join(supported_types)}"
            raise CompileError(msg)

        return type1


class LogicalNotExpression(Expression):
    def __init__(self, expression1: Expression):
        super().__init__()
        self.expression1: Expression = expression1

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        type1 = self.expression1.compile(ncs, root, block)

        if type1 == DynamicDataType.INT:
            ncs.add(NCSInstructionType.NOTI)
        else:
            msg = f"Logical NOT requires integer operand, got {type1.builtin.name.lower()}\n  Note: In NWScript, only int types can be used in logical operations"
            raise CompileError(msg)

        return DynamicDataType.INT


class BitwiseNotExpression(Expression):
    def __init__(self, expression1: Expression):
        super().__init__()
        self.expression1: Expression = expression1

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        type1 = self.expression1.compile(ncs, root, block)

        if type1 == DynamicDataType.INT:
            ncs.add(NCSInstructionType.COMPI)
        else:
            msg = f"Bitwise NOT (~) requires integer operand, got {type1.builtin.name.lower()}\n  Note: Bitwise operations only work on int types"
            raise CompileError(msg)

        return type1


# region Expressions: Assignment
class Assignment(Expression):
    def __init__(
        self,
        field_access: FieldAccess,
        value: Expression,
        *,
        allow_const: bool = False,
    ):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value
        self.allow_const = allow_const

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        variable_type = self.expression.compile(ncs, root, block)

        # Get the variable location; get_scoped accounts for the current transient expression value
        is_global, expression_type, stack_index, is_const = self.field_access.get_scoped(
            block,
            root,
        )

        if is_const and not self.allow_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)

        instruction_type = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        # get_scoped() already accounts for the transient expression result, so stack_index
        # points to the correct variable location

        if variable_type != expression_type:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Type mismatch in assignment to '{var_name}'\n  Variable type: {expression_type.builtin.name}\n  Expression type: {variable_type.builtin.name}"
            raise CompileError(msg)

        # Copy the value that the expression has already been placed on the stack to where the identifiers position is
        ncs.instructions.append(
            NCSInstruction(instruction_type, [stack_index, expression_type.size(root)]),
        )

        # Leave the assignment result on the stack; the enclosing expression consumer owns cleanup.

        return variable_type


class AdditionAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
    ) -> DynamicDataType:
        # Copy the variable to the top of the stack
        is_global, variable_type, stack_index, is_const = self.field_access.get_scoped(
            block,
            root,
        )
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expresion_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expresion_type == DynamicDataType.INT:
            arthimetic_instruction = NCSInstructionType.ADDII
        elif variable_type == DynamicDataType.FLOAT and expresion_type == DynamicDataType.FLOAT:
            arthimetic_instruction = NCSInstructionType.ADDFF
        elif variable_type == DynamicDataType.FLOAT and expresion_type == DynamicDataType.INT:
            arthimetic_instruction = NCSInstructionType.ADDFI
        elif variable_type == DynamicDataType.STRING and expresion_type == DynamicDataType.STRING:
            arthimetic_instruction = NCSInstructionType.ADDSS
        elif variable_type == DynamicDataType.VECTOR and expresion_type == DynamicDataType.VECTOR:
            arthimetic_instruction = NCSInstructionType.ADDVV
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in += operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expresion_type.builtin.name}\n"
                f"  Supported: int+=int, float+=float/int, string+=string, vector+=vector"
            )
            raise CompileError(msg)

        # Add the expression and our temp variable copy together
        ncs.add(arthimetic_instruction, args=[])

        # Copy the result to the original variable in the stack
        # The arithmetic operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Arithmetic operation consumed variable copy and expression (2 values), left result (1 value)
        # Result is still on stack (copied to variable location but also remains on top for ExpressionStatement)
        # The explicit stack context mirrors the VM transformation: two operands become one result.
        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class SubtractionAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        isglobal, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if isglobal else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expresion_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expresion_type == DynamicDataType.INT:
            arthimetic_instruction = NCSInstructionType.SUBII
        elif variable_type == DynamicDataType.FLOAT and expresion_type == DynamicDataType.FLOAT:
            arthimetic_instruction = NCSInstructionType.SUBFF
        elif variable_type == DynamicDataType.FLOAT and expresion_type == DynamicDataType.INT:
            arthimetic_instruction = NCSInstructionType.SUBFI
        elif variable_type == DynamicDataType.VECTOR and expresion_type == DynamicDataType.VECTOR:
            arthimetic_instruction = NCSInstructionType.SUBVV
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in -= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expresion_type.builtin.name}\n"
                f"  Supported: int-=int, float-=float/int, vector-=vector"
            )
            raise CompileError(msg)

        # Subtract the expression from our temp variable copy
        ncs.add(arthimetic_instruction)

        # Copy the result to the original variable in the stack
        # The arithmetic operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if isglobal else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if isglobal else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Arithmetic operation consumed variable copy and expression (2 values), left result (1 value)
        # Result is still on stack (copied to variable location but also remains on top for ExpressionStatement)
        # The explicit stack context mirrors the VM transformation: two operands become one result.
        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class MultiplicationAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        isglobal, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if isglobal else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expresion_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expresion_type == DynamicDataType.INT:
            arthimetic_instruction = NCSInstructionType.MULII
        elif variable_type == DynamicDataType.FLOAT and expresion_type == DynamicDataType.FLOAT:
            arthimetic_instruction = NCSInstructionType.MULFF
        elif variable_type == DynamicDataType.FLOAT and expresion_type == DynamicDataType.INT:
            arthimetic_instruction = NCSInstructionType.MULFI
        elif variable_type == DynamicDataType.VECTOR and expresion_type == DynamicDataType.FLOAT:
            arthimetic_instruction = NCSInstructionType.MULVF
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in *= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expresion_type.builtin.name}\n"
                f"  Supported: int*=int, float*=float/int, vector*=float"
            )
            raise CompileError(msg)

        # Multiply the temp variable copy by the expression
        ncs.add(arthimetic_instruction)

        # Copy the result to the original variable in the stack
        # The arithmetic operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if isglobal else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if isglobal else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Arithmetic operation consumed variable copy and expression (2 values), left result (1 value)
        # Result is still on stack (copied to variable location but also remains on top for ExpressionStatement)
        # The explicit stack context mirrors the VM transformation: two operands become one result.
        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class DivisionAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        isglobal, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if isglobal else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expresion_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expresion_type == DynamicDataType.INT:
            arthimetic_instruction = NCSInstructionType.DIVII
        elif variable_type == DynamicDataType.FLOAT and expresion_type == DynamicDataType.FLOAT:
            arthimetic_instruction = NCSInstructionType.DIVFF
        elif variable_type == DynamicDataType.FLOAT and expresion_type == DynamicDataType.INT:
            arthimetic_instruction = NCSInstructionType.DIVFI
        elif variable_type == DynamicDataType.VECTOR and expresion_type == DynamicDataType.FLOAT:
            arthimetic_instruction = NCSInstructionType.DIVVF
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in /= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expresion_type.builtin.name}\n"
                f"  Supported: int/=int, float/=float/int, vector/=float"
            )
            raise CompileError(msg)

        # Divide the temp variable copy by the expression
        ncs.add(arthimetic_instruction)

        # Copy the result to the original variable in the stack
        # The arithmetic operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if isglobal else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if isglobal else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Arithmetic operation consumed variable copy and expression (2 values), left result (1 value)
        # Result is still on stack (copied to variable location but also remains on top for ExpressionStatement)
        # The explicit stack context mirrors the VM transformation: two operands become one result.
        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class ModuloAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        isglobal, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if isglobal else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expresion_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expresion_type == DynamicDataType.INT:
            arthimetic_instruction = NCSInstructionType.MODII
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in %= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expresion_type.builtin.name}\n"
                f"  Supported: int%=int"
            )
            raise CompileError(msg)

        # Apply modulo operation
        ncs.add(arthimetic_instruction)

        # Copy the result to the original variable in the stack
        # The arithmetic operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if isglobal else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if isglobal else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Arithmetic operation consumed variable copy and expression (2 values), left result (1 value)
        # Result is still on stack (copied to variable location but also remains on top for ExpressionStatement)
        # The explicit stack context mirrors the VM transformation: two operands become one result.
        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class BitwiseAndAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        is_global, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expression_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expression_type == DynamicDataType.INT:
            bitwise_instruction = NCSInstructionType.BOOLANDII
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in &= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expression_type.builtin.name}\n"
                f"  Supported: int&=int"
            )
            raise CompileError(msg)

        # Apply the bitwise AND operation
        ncs.add(bitwise_instruction, args=[])

        # Copy the result to the original variable in the stack
        # The bitwise operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Bitwise operation consumed variable copy and expression (2 values), left result (1 value)
        # The explicit stack context records that both operands became one result.
        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class BitwiseOrAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        is_global, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expression_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expression_type == DynamicDataType.INT:
            bitwise_instruction = NCSInstructionType.INCORII
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in |= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expression_type.builtin.name}\n"
                f"  Supported: int|=int"
            )
            raise CompileError(msg)

        # Apply the bitwise OR operation
        ncs.add(bitwise_instruction, args=[])

        # Copy the result to the original variable in the stack
        # The bitwise operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Bitwise operation consumed variable copy and expression (2 values), left result (1 value)
        # The explicit stack context records that both operands became one result.
        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class BitwiseXorAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        is_global, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expression_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expression_type == DynamicDataType.INT:
            bitwise_instruction = NCSInstructionType.EXCORII
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in ^= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expression_type.builtin.name}\n"
                f"  Supported: int^=int"
            )
            raise CompileError(msg)

        # Apply the bitwise XOR operation
        ncs.add(bitwise_instruction, args=[])

        # Copy the result to the original variable in the stack
        # The bitwise operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Bitwise operation consumed variable copy and expression (2 values), left result (1 value)
        # The explicit stack context records that both operands became one result.
        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class BitwiseLeftAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        is_global, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expression_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expression_type == DynamicDataType.INT:
            bitwise_instruction = NCSInstructionType.SHLEFTII
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in <<= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expression_type.builtin.name}\n"
                f"  Supported: int<<=int"
            )
            raise CompileError(msg)

        # Apply the bitwise left shift operation
        ncs.add(bitwise_instruction, args=[])

        # Copy the result to the original variable in the stack
        # The bitwise operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Bitwise operation consumed variable copy and expression (2 values), left result (1 value)
        # The explicit stack context records that both operands became one result.
        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class BitwiseRightAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        is_global, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expression_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expression_type == DynamicDataType.INT:
            bitwise_instruction = NCSInstructionType.SHRIGHTII
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in >>= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expression_type.builtin.name}\n"
                f"  Supported: int>>=int"
            )
            raise CompileError(msg)

        # Apply the bitwise right shift operation
        ncs.add(bitwise_instruction, args=[])

        # Copy the result to the original variable in the stack
        # The bitwise operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Bitwise operation consumed variable copy and expression (2 values), left result (1 value)
        # The explicit stack context records that both operands became one result.
        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


class BitwiseUnsignedRightAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        # Copy the variable to the top of the stack
        is_global, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot assign to const variable '{var_name}'"
            raise CompileError(msg)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        # Compile the right-hand side; the expression contract tracks its result.
        expression_type = self.expression.compile(ncs, root, block)

        # Determine what instruction to apply to the two values
        if variable_type == DynamicDataType.INT and expression_type == DynamicDataType.INT:
            bitwise_instruction = NCSInstructionType.USHRIGHTII
        else:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = (
                f"Type mismatch in >>>= operation on '{var_name}'\n"
                f"  Variable type: {variable_type.builtin.name}\n"
                f"  Expression type: {expression_type.builtin.name}\n"
                f"  Supported: int>>>=int"
            )
            raise CompileError(msg)

        # Apply the unsigned bitwise right shift operation
        ncs.add(bitwise_instruction, args=[])

        # Copy the result to the original variable in the stack
        # The bitwise operation consumed both operands and left the result on stack
        # After CPDOWNSP, the result is still on stack (for ExpressionStatement to clean up)
        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        # Result (variable_type size) is on stack; offset to original variable accounts for this
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        # Bitwise operation consumed variable copy and expression (2 values), left result (1 value)
        # The explicit stack context records that both operands became one result.
        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        # Return variable_type (the result type) so ExpressionStatement knows what size to clean up
        return variable_type


# endregion


# region Statements
class EmptyStatement(Statement):
    def __init__(self):
        super().__init__()

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ) -> DynamicDataType:
        return DynamicDataType.VOID


class NopStatement(Statement):
    def __init__(self, string: str):
        super().__init__()
        self.string = string

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ) -> DynamicDataType:
        ncs.add(NCSInstructionType.NOP, args=[self.string])
        return DynamicDataType.VOID


class ExpressionStatement(Statement):
    def __init__(self, expression: Expression):
        super().__init__()
        self.expression: Expression = expression

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        expression_type = self.expression.compile(ncs, root, block)
        if expression_type != DynamicDataType.VOID:
            expression_size = expression_type.size(root)
            ncs.add(NCSInstructionType.MOVSP, args=[-expression_size])
            block.context.stack.consume(expression_size)


class DeclarationStatement(Statement):
    def __init__(
        self,
        data_type: DynamicDataType,
        declarators: list[VariableDeclarator | VariableInitializer],
        is_const: bool = False,
    ):
        super().__init__()
        self.data_type: DynamicDataType = data_type
        self.declarators: list[VariableDeclarator | VariableInitializer] = declarators
        self.is_const: bool = is_const

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        if self.is_const:
            raise ValueError(
                "Internal compiler error: local const declaration reached emission "
                "after semantic validation"
            )
        for declarator in self.declarators:
            declarator.compile(ncs, root, block, self.data_type, False)


class VariableDeclarator:
    def __init__(self, identifier: Identifier):
        self.identifier: Identifier = identifier

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        data_type: DynamicDataType,
        is_const: bool = False,
    ):
        if data_type.builtin == DataType.INT:
            ncs.add(NCSInstructionType.RSADDI)
        elif data_type.builtin == DataType.FLOAT:
            ncs.add(NCSInstructionType.RSADDF)
        elif data_type.builtin == DataType.STRING:
            ncs.add(NCSInstructionType.RSADDS)
        elif data_type.builtin == DataType.OBJECT:
            ncs.add(NCSInstructionType.RSADDO)
        elif data_type.builtin == DataType.EVENT:
            ncs.add(NCSInstructionType.RSADDEVT)
        elif data_type.builtin == DataType.LOCATION:
            ncs.add(NCSInstructionType.RSADDLOC)
        elif data_type.builtin == DataType.TALENT:
            ncs.add(NCSInstructionType.RSADDTAL)
        elif data_type.builtin == DataType.EFFECT:
            ncs.add(NCSInstructionType.RSADDEFF)
        elif data_type.builtin == DataType.VECTOR:
            ncs.add(NCSInstructionType.RSADDF)
            ncs.add(NCSInstructionType.RSADDF)
            ncs.add(NCSInstructionType.RSADDF)
        elif data_type.builtin == DataType.STRUCT:
            struct_name = data_type._struct  # noqa: SLF001
            if struct_name is not None and struct_name in root.struct_map:
                root.struct_map[struct_name].initialize(ncs, root)
            else:
                raise ValueError(
                    f"Internal compiler error: unknown struct type for local '{self.identifier}' "
                    "reached emission"
                )
        elif data_type.builtin == DataType.VOID:
            raise ValueError(
                f"Internal compiler error: void local '{self.identifier}' reached emission"
            )
        else:
            raise ValueError(
                f"Internal compiler error: unsupported local type "
                f"'{data_type.builtin.name}' reached emission"
            )

        block.add_scoped(self.identifier, data_type, is_const)


class VariableInitializer:
    def __init__(self, identifier: Identifier, expression: Expression):
        self.identifier: Identifier = identifier
        self.expression: Expression = expression

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        data_type: DynamicDataType,
        is_const: bool = False,
    ):
        declarator = VariableDeclarator(self.identifier)
        declarator.compile(ncs, root, block, data_type, is_const)

        # Initializers use normal assignment lowering but may write the freshly
        # declared const slot exactly once. The expression contract guarantees one
        # result value that we discard after copying it into the variable.
        assignment = Assignment(
            FieldAccess([self.identifier]),
            self.expression,
            allow_const=True,
        )
        result_type = assignment.compile(ncs, root, block)
        result_size = result_type.size(root)
        ncs.add(NCSInstructionType.MOVSP, args=[-result_size])
        block.context.stack.consume(result_size)


class ConditionalBlock(Statement):
    def __init__(
        self,
        if_block: ConditionAndBlock,
        else_if_blocks: list[ConditionAndBlock],
        else_block: CodeBlock,
    ):
        super().__init__()
        self.if_blocks: list[ConditionAndBlock] = [if_block, *else_if_blocks]
        self.else_block: CodeBlock | None = else_block

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        """Compile an if/else chain, pruning branches with constant conditions.

        This mirrors BioWare's dead-branch optimization: only conditions that
        fold to an integer constant are removed. Runtime conditions retain the
        normal JZ/JMP control-flow structure.
        """
        end_label = NCSInstruction(NCSInstructionType.NOP, args=[])
        needs_end_label = False

        for condition_and_block in self.if_blocks:
            constant = condition_and_block.condition.constant_value(
                root,
                block.context.semantic.source_origin,
            )
            if constant is not None:
                if constant.datatype != DataType.INT:
                    raise ValueError(
                        "Internal compiler error: non-int constant condition reached emission"
                    )
                if int(constant.value) == 0:
                    # Provably dead branch: emit neither the condition nor body.
                    continue

                # Provably true. Any remaining else-if/else branches are dead.
                condition_and_block.block.compile(
                    ncs,
                    root,
                    block,
                    return_instruction,
                    break_instruction,
                    continue_instruction,
                )
                if needs_end_label:
                    ncs.instructions.append(end_label)
                return

            next_label = NCSInstruction(NCSInstructionType.NOP, args=[])
            condition_type = condition_and_block.condition.compile(ncs, root, block)
            if condition_type != DynamicDataType.INT:
                raise ValueError(
                    "Internal compiler error: non-int condition reached emission"
                )

            ncs.add(NCSInstructionType.JZ, jump=next_label)
            block.context.stack.consume(condition_type.size(root))

            condition_and_block.block.compile(
                ncs,
                root,
                block,
                return_instruction,
                break_instruction,
                continue_instruction,
            )
            ncs.add(NCSInstructionType.JMP, jump=end_label)
            needs_end_label = True
            ncs.instructions.append(next_label)

        if self.else_block is not None:
            self.else_block.compile(
                ncs,
                root,
                block,
                return_instruction,
                break_instruction,
                continue_instruction,
            )

        if needs_end_label:
            ncs.instructions.append(end_label)


class ConditionAndBlock:
    def __init__(self, condition: Expression, block: CodeBlock):
        self.condition: Expression = condition
        self.block: CodeBlock = block


class ReturnStatement(Statement):
    def __init__(self, expression: Expression | None = None):
        super().__init__()
        self.expression: Expression | None = expression

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ) -> DynamicDataType:
        expected_type = block.context.semantic.return_type
        function_name = block.context.semantic.function_name or "<function>"
        entry_transient_bytes = block.context.stack.snapshot()

        if self.expression is None:
            return_type = DynamicDataType.VOID
            if expected_type is not None and expected_type != DynamicDataType.VOID:
                raise ValueError(
                    f"Internal compiler error: bare return in non-void function '{function_name}'"
                )
        else:
            if expected_type == DynamicDataType.VOID:
                raise ValueError(
                    f"Internal compiler error: value return in void function '{function_name}'"
                )
            return_type = self.expression.compile(ncs, root, block)
            if expected_type is not None and return_type != expected_type:
                raise ValueError(
                    f"Internal compiler error: return type mismatch reached emission in "
                    f"'{function_name}'"
                )

        scope_size = block.full_scope_size(root)
        if return_type != DynamicDataType.VOID:
            return_size = return_type.size(root)
            ncs.add(
                NCSInstructionType.CPDOWNSP,
                args=[-scope_size - entry_transient_bytes - return_size * 2, return_size],
            )
            ncs.add(NCSInstructionType.MOVSP, args=[-return_size])
            block.context.stack.consume(return_size)

        cleanup_size = scope_size + entry_transient_bytes
        if cleanup_size != 0:
            ncs.add(NCSInstructionType.MOVSP, args=[-cleanup_size])
        # Emission has terminated this runtime path, but compilation may continue
        # at another control-flow label (for example a later switch case). Keep the
        # compile-time transient depth at the return statement's entry state.
        block.context.stack.restore(entry_transient_bytes)
        ncs.add(NCSInstructionType.JMP, jump=return_instruction)
        return return_type


class WhileLoopBlock(Statement):
    def __init__(self, condition: Expression, block: CodeBlock):
        super().__init__()
        self.condition: Expression = condition
        self.block: CodeBlock = block

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        loopstart = ncs.add(NCSInstructionType.NOP, args=[])
        loopend = NCSInstruction(NCSInstructionType.NOP, args=[])
        target_depth = block.stack_depth(root)

        condition_type = self.condition.compile(ncs, root, block)
        if condition_type != DynamicDataType.INT:
            raise ValueError(
                "Internal compiler error: non-int while condition reached emission"
            )

        ncs.add(NCSInstructionType.JZ, jump=loopend)
        block.context.stack.consume(condition_type.size(root))

        target = ControlFlowTarget(
            ControlKeyword.WHILE,
            loopend,
            target_depth,
            loopstart,
            target_depth,
        )
        block.context.control.push(target)
        try:
            self.block.compile(ncs, root, block, return_instruction, loopend, loopstart)
        finally:
            block.context.control.pop(target)
        ncs.add(NCSInstructionType.JMP, jump=loopstart)
        ncs.instructions.append(loopend)


class DoWhileLoopBlock(Statement):
    def __init__(self, condition: Expression, block: CodeBlock):
        super().__init__()
        self.condition: Expression = condition
        self.block: CodeBlock = block

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        loopstart = ncs.add(NCSInstructionType.NOP, args=[])
        conditionstart = NCSInstruction(NCSInstructionType.NOP, args=[])
        loopend = NCSInstruction(NCSInstructionType.NOP, args=[])
        target_depth = block.stack_depth(root)

        target = ControlFlowTarget(
            ControlKeyword.DO,
            loopend,
            target_depth,
            conditionstart,
            target_depth,
        )
        block.context.control.push(target)
        try:
            self.block.compile(
                ncs,
                root,
                block,
                return_instruction,
                loopend,
                conditionstart,
            )
        finally:
            block.context.control.pop(target)

        ncs.instructions.append(conditionstart)
        condition_type = self.condition.compile(ncs, root, block)
        if condition_type != DynamicDataType.INT:
            raise ValueError(
                "Internal compiler error: non-int do-while condition reached emission"
            )

        ncs.add(NCSInstructionType.JZ, jump=loopend)
        block.context.stack.consume(condition_type.size(root))
        ncs.add(NCSInstructionType.JMP, jump=loopstart)
        ncs.instructions.append(loopend)


class ForLoopBlock(Statement):
    def __init__(
        self,
        initial: Expression | Statement | None,
        condition: Expression,
        iteration: Expression | None,
        block: CodeBlock,
    ):
        super().__init__()
        self.initial: Expression | Statement | None = initial
        self.condition: Expression = condition
        self.iteration: Expression | None = iteration
        self.block: CodeBlock = block

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        if self.initial is not None:
            if isinstance(self.initial, Statement):
                # For declaration statements, compile them directly
                self.initial.compile(
                    ncs, root, block, return_instruction, break_instruction, continue_instruction
                )
            else:
                initial_type = self.initial.compile(ncs, root, block)
                initial_size = initial_type.size(root)
                ncs.add(NCSInstructionType.MOVSP, args=[-initial_size])
                block.context.stack.consume(initial_size)

        loopstart = ncs.add(NCSInstructionType.NOP, args=[])
        updatestart = NCSInstruction(NCSInstructionType.NOP, args=[])
        loopend = NCSInstruction(NCSInstructionType.NOP, args=[])

        target_depth = block.stack_depth(root)
        condition_type = self.condition.compile(ncs, root, block)
        if condition_type != DynamicDataType.INT:
            raise ValueError(
                "Internal compiler error: non-int for condition reached emission"
            )

        ncs.add(NCSInstructionType.JZ, jump=loopend)
        block.context.stack.consume(condition_type.size(root))

        target = ControlFlowTarget(
            ControlKeyword.FOR,
            loopend,
            target_depth,
            updatestart,
            target_depth,
        )
        block.context.control.push(target)
        try:
            self.block.compile(ncs, root, block, return_instruction, loopend, updatestart)
        finally:
            block.context.control.pop(target)

        ncs.instructions.append(updatestart)
        if self.iteration is not None:
            iteration_type = self.iteration.compile(ncs, root, block)
            iteration_size = iteration_type.size(root)
            ncs.add(NCSInstructionType.MOVSP, args=[-iteration_size])
            block.context.stack.consume(iteration_size)

        ncs.add(NCSInstructionType.JMP, jump=loopstart)
        ncs.instructions.append(loopend)


class ScopedBlock(Statement):
    def __init__(self, block: CodeBlock):
        super().__init__()
        self.block: CodeBlock = block

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        self.block.compile(
            ncs,
            root,
            block,
            return_instruction,
            break_instruction,
            continue_instruction,
        )


# endregion


class BreakStatement(Statement):
    def __init__(self):
        super().__init__()

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        target = block.context.control.break_target()
        if target is None:
            raise ValueError(
                "Internal compiler error: break without control-flow target reached emission"
            )
        unwind_size = block.unwind_size_to(root, target.break_stack_depth)
        if unwind_size:
            ncs.add(NCSInstructionType.MOVSP, args=[-unwind_size])
        ncs.add(NCSInstructionType.JMP, jump=target.break_instruction)


class ContinueStatement(Statement):
    def __init__(self):
        super().__init__()

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        target = block.context.control.continue_target()
        if target is None or target.continue_instruction is None or target.continue_stack_depth is None:
            raise ValueError(
                "Internal compiler error: continue without loop target reached emission"
            )
        unwind_size = block.unwind_size_to(root, target.continue_stack_depth)
        if unwind_size:
            ncs.add(NCSInstructionType.MOVSP, args=[-unwind_size])
        ncs.add(NCSInstructionType.JMP, jump=target.continue_instruction)


class PrefixIncrementExpression(Expression):
    def __init__(self, field_access: FieldAccess):
        self.field_access: FieldAccess = field_access

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        variable_type = self.field_access.compile(ncs, root, block)

        if variable_type != DynamicDataType.INT:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Increment operator (++) requires integer variable, got {variable_type.builtin.name.lower()}\n  Variable: {var_name}"
            raise CompileError(msg)

        isglobal, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot increment const variable '{var_name}'"
            raise CompileError(msg)
        ncs.add(NCSInstructionType.INCISP, args=[-4])

        if isglobal:
            ncs.add(
                NCSInstructionType.CPDOWNBP,
                args=[stack_index, variable_type.size(root)],
            )
        else:
            ncs.add(
                NCSInstructionType.CPDOWNSP,
                args=[stack_index, variable_type.size(root)],
            )

        return variable_type


class PostfixIncrementExpression(Expression):
    def __init__(self, field_access: FieldAccess):
        self.field_access: FieldAccess = field_access

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        variable_type = self.field_access.compile(ncs, root, block)

        if variable_type != DynamicDataType.INT:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Increment operator (++) requires integer variable, got {variable_type.builtin.name.lower()}\n  Variable: {var_name}"
            raise CompileError(msg)

        isglobal, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot increment const variable '{var_name}'"
            raise CompileError(msg)
        if isglobal:
            ncs.add(NCSInstructionType.INCIBP, args=[stack_index])
        else:
            ncs.add(NCSInstructionType.INCISP, args=[stack_index])

        return variable_type


class PrefixDecrementExpression(Expression):
    def __init__(self, field_access: FieldAccess):
        self.field_access: FieldAccess = field_access

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        variable_type = self.field_access.compile(ncs, root, block)

        if variable_type != DynamicDataType.INT:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Decrement operator (--) requires integer variable, got {variable_type.builtin.name.lower()}\n  Variable: {var_name}"
            raise CompileError(msg)

        isglobal, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot decrement const variable '{var_name}'"
            raise CompileError(msg)
        ncs.add(NCSInstructionType.DECISP, args=[-4])

        if isglobal:
            ncs.add(
                NCSInstructionType.CPDOWNBP,
                args=[stack_index, variable_type.size(root)],
            )
        else:
            ncs.add(
                NCSInstructionType.CPDOWNSP,
                args=[stack_index, variable_type.size(root)],
            )

        return variable_type


class PostfixDecrementExpression(Expression):
    def __init__(self, field_access: FieldAccess):
        self.field_access: FieldAccess = field_access

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        variable_type = self.field_access.compile(ncs, root, block)

        if variable_type != DynamicDataType.INT:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Decrement operator (--) requires integer variable, got {variable_type.builtin.name.lower()}\n  Variable: {var_name}"
            raise CompileError(msg)

        isglobal, variable_type, stack_index, is_const = self.field_access.get_scoped(block, root)
        if is_const:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Cannot decrement const variable '{var_name}'"
            raise CompileError(msg)
        if isglobal:
            ncs.add(NCSInstructionType.DECIBP, args=[stack_index])
        else:
            ncs.add(NCSInstructionType.DECISP, args=[stack_index])

        return variable_type


# region Switch
class SwitchStatement(Statement):
    def __init__(self, expression: Expression, blocks: list[SwitchBlock]):
        super().__init__()
        self.expression: Expression = expression
        self.blocks: list[SwitchBlock] = blocks

        self.real_block: CodeBlock = CodeBlock()

    def _validate_labels(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None,
    ) -> tuple[list[tuple[SwitchBlock, int]], SwitchBlock | None]:
        """Resolve BioWare switch labels before emitting any switch bytecode.

        NWScript switch labels are compile-time integer constants.  Resolving
        them up front also lets us reject duplicate case values and duplicate
        default labels deterministically, including duplicates that only become
        equal after constant folding.
        """
        cases: list[tuple[SwitchBlock, int]] = []
        case_values: set[int] = set()
        default_block: SwitchBlock | None = None

        for switchblock in self.blocks:
            for label in switchblock.labels:
                if isinstance(label, DefaultSwitchLabel):
                    if default_block is not None:
                        raise ValueError(
                            "Internal compiler error: duplicate default label reached emission"
                        )
                    default_block = switchblock
                    continue

                if not isinstance(label, ExpressionSwitchLabel):
                    raise ValueError(
                        "Internal compiler error: unsupported switch label reached emission: "
                        f"{type(label).__name__}"
                    )

                constant = label.expression.constant_value(root, source_origin)
                if constant is None or constant.datatype != DataType.INT:
                    raise ValueError(
                        "Internal compiler error: non-constant/non-int switch label reached emission"
                    )

                value = _int32(int(constant.value))
                if value in case_values:
                    raise ValueError(
                        f"Internal compiler error: duplicate switch case value {value} reached emission"
                    )
                case_values.add(value)
                cases.append((switchblock, value))

        return cases, default_block

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        # Validate labels before emitting the switch expression or body.
        cases, default_block = self._validate_labels(
            root,
            block.context.semantic.source_origin,
        )

        parent_block = block
        self.real_block._parent = parent_block  # noqa: SLF001
        self.real_block._context = parent_block.context  # noqa: SLF001
        block = self.real_block

        expression_type = self.expression.compile(ncs, root, block)
        if expression_type != DynamicDataType.INT:
            raise ValueError(
                "Internal compiler error: non-int switch expression reached emission"
            )

        end_of_switch = NCSInstruction(NCSInstructionType.NOP, args=[])
        switch_stack_depth = block.stack_depth(root)
        target = ControlFlowTarget(
            ControlKeyword.SWITCH,
            end_of_switch,
            switch_stack_depth,
        )

        # Compile the body once. Blocks remain sequential so normal fall-through
        # semantics are preserved. A labeled block must be reachable with exactly
        # the switch-entry stack layout; otherwise a case jump would skip locals.
        tempncs = NCS()
        switchblock_to_instruction: dict[SwitchBlock, NCSInstruction] = {}
        block.context.control.push(target)
        try:
            for switchblock in self.blocks:
                if switchblock.labels and block.stack_depth(root) != switch_stack_depth:
                    raise ValueError(
                        "Internal compiler error: switch label stack depth changed after "
                        "semantic validation"
                    )
                switchblock_start = tempncs.add(NCSInstructionType.NOP, args=[])
                switchblock_to_instruction[switchblock] = switchblock_start
                for statement in switchblock.block:
                    statement.compile(
                        tempncs,
                        root,
                        block,
                        return_instruction,
                        end_of_switch,
                        continue_instruction,
                    )
        finally:
            block.context.control.pop(target)

        # Normal body fall-through owns any switch-scope locals it allocated. A
        # break/no-match jump targets end_of_switch directly and therefore skips
        # this cleanup because those paths never have those locals (or already
        # unwound them explicitly).
        switch_scope_size = block.scope_size(root)
        if switch_scope_size:
            tempncs.add(NCSInstructionType.MOVSP, args=[-switch_scope_size])

        # Test every case first.  Case expressions are compile-time constants,
        # so no case-label expression is evaluated at runtime.
        for switchblock, case_value in cases:
            ncs.add(NCSInstructionType.CPTOPSP, args=[-4, 4])
            ncs.add(NCSInstructionType.CONSTI, args=[case_value])
            ncs.add(NCSInstructionType.EQUALII, args=[])
            ncs.add(NCSInstructionType.JNZ, jump=switchblock_to_instruction[switchblock])

        # BioWare semantics: default is selected only after every case fails,
        # irrespective of where the default label appears textually.
        if default_block is not None:
            ncs.add(NCSInstructionType.JMP, jump=switchblock_to_instruction[default_block])
        else:
            ncs.add(NCSInstructionType.JMP, jump=end_of_switch)

        ncs.merge(tempncs)
        ncs.instructions.append(end_of_switch)

        # All switch exits converge with only the discriminant still transient.
        ncs.add(NCSInstructionType.MOVSP, args=[-expression_type.size(root)])
        block.context.stack.consume(expression_type.size(root))


class SwitchBlock:
    def __init__(self, labels: list[SwitchLabel], block: list[Statement]):
        self.labels: list[SwitchLabel] = labels
        self.block: list[Statement] = block


class SwitchLabel(ABC):
    """Marker base class for parsed switch labels."""


class ExpressionSwitchLabel(SwitchLabel):
    def __init__(self, expression: Expression):
        self.expression: Expression = expression


class DefaultSwitchLabel(SwitchLabel):
    pass


def _statement_returns_on_all_paths(statement: Statement | CodeBlock) -> bool:
    """Return whether BioWare's conservative analysis proves this statement returns."""
    if isinstance(statement, CodeBlock):
        return statement.returns_on_all_paths()
    if isinstance(statement, ReturnStatement):
        return True
    if isinstance(statement, ScopedBlock):
        return statement.block.returns_on_all_paths()
    if isinstance(statement, ConditionalBlock):
        if statement.else_block is None:
            return False
        return all(
            condition_and_block.block.returns_on_all_paths()
            for condition_and_block in statement.if_blocks
        ) and statement.else_block.returns_on_all_paths()
    return False


# endregion


class DynamicDataType:
    INT: DynamicDataType
    STRING: DynamicDataType
    FLOAT: DynamicDataType
    OBJECT: DynamicDataType
    VECTOR: DynamicDataType
    EVENT: DynamicDataType
    TALENT: DynamicDataType
    LOCATION: DynamicDataType
    EFFECT: DynamicDataType
    VOID: DynamicDataType

    def __init__(self, datatype: DataType, struct_name: str | None = None):
        self.builtin: DataType = datatype
        self._struct: str | None = struct_name

    @property
    def struct_name(self) -> str | None:
        """User-struct name when ``builtin`` is :class:`DataType.STRUCT`."""
        return self._struct

    def __eq__(self, other: DynamicDataType | DataType | object) -> bool:
        if self is other:
            return True
        if isinstance(other, DynamicDataType):
            if self.builtin == other.builtin:
                return self.builtin != DataType.STRUCT or (
                    self.builtin == DataType.STRUCT and self._struct == other._struct
                )
            return False
        if isinstance(other, DataType):
            return self.builtin == other and self.builtin != DataType.STRUCT
        return NotImplemented  # type: ignore[no-any-return]

    def __hash__(self) -> int:
        return hash(self.builtin) ^ hash(self._struct)

    def __repr__(self) -> str:
        return f"DynamicDataType(builtin={self.builtin}({self.builtin.name.lower()}), struct={self._struct})"

    def size(self, root: CodeRoot) -> int:
        if self.builtin == DataType.STRUCT:
            if self._struct is None:
                raise CompileError("Struct type has no name")  # noqa: B904
            return root.struct_map[self._struct].size(root)
        return self.builtin.size()


DynamicDataType.INT = DynamicDataType(DataType.INT)
DynamicDataType.STRING = DynamicDataType(DataType.STRING)
DynamicDataType.FLOAT = DynamicDataType(DataType.FLOAT)
DynamicDataType.OBJECT = DynamicDataType(DataType.OBJECT)
DynamicDataType.VECTOR = DynamicDataType(DataType.VECTOR)
DynamicDataType.VOID = DynamicDataType(DataType.VOID)
DynamicDataType.EVENT = DynamicDataType(DataType.EVENT)
DynamicDataType.TALENT = DynamicDataType(DataType.TALENT)
DynamicDataType.LOCATION = DynamicDataType(DataType.LOCATION)
DynamicDataType.EFFECT = DynamicDataType(DataType.EFFECT)
