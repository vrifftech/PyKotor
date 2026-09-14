"""AST nodes, symbol resolution, and NCS instruction emission."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from utility.system.path import Path
from typing import TYPE_CHECKING, Iterator, NamedTuple, cast

from pykotor.common.script import DataType
from pykotor.resource.formats.ncs import NCS, NCSInstruction, NCSInstructionType
from pykotor.resource.formats.ncs.compiler.numeric import (
    canonicalize_kotor_float_constant,
    float32 as _float32,
    int32 as _int32,
)
from pykotor.resource.formats.ncs.compiler.source import CompileError, SourceLocation
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


class EntryPointError(CompileError):
    """Raised when script has no valid entry point (main or StartingConditional)."""


class MissingIncludeError(CompileError):
    """Raised when a #include file cannot be found."""


DEFAULT_MAX_INCLUDE_DEPTH = 16
MAX_FUNCTION_PARAMETERS = 32


class IncludeContext:
    """Track active and completed includes; the root counts toward the depth limit."""

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

        # The root counts as the first file in the include-depth limit.
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
    """A typed predefined constant."""

    datatype: DataType
    value: int | float | str


@dataclass(frozen=True)
class SourceOrigin:
    """Declaration order and include depth after source expansion."""

    file_level: int
    source_order: int


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


def emit_value_reservation(ncs: NCS, root: CodeRoot, datatype: DynamicDataType) -> int:
    """Reserve typed stack cells for a return value or aggregate member."""
    instructions = {
        DataType.INT: NCSInstructionType.RSADDI,
        DataType.FLOAT: NCSInstructionType.RSADDF,
        DataType.STRING: NCSInstructionType.RSADDS,
        DataType.OBJECT: NCSInstructionType.RSADDO,
        DataType.EVENT: NCSInstructionType.RSADDEVT,
        DataType.LOCATION: NCSInstructionType.RSADDLOC,
        DataType.TALENT: NCSInstructionType.RSADDTAL,
        DataType.EFFECT: NCSInstructionType.RSADDEFF,
    }
    if datatype.builtin == DataType.VOID:
        return 0
    if datatype.builtin == DataType.VECTOR:
        for _ in range(3):
            ncs.add(NCSInstructionType.RSADDF)
    elif datatype.builtin == DataType.STRUCT:
        struct_type = root.struct_map.get(datatype.struct_name)
        if struct_type is None:
            raise CompileError(f"Unknown struct type '{datatype.struct_name}' for value storage")
        struct_type.initialize(ncs, root)
    elif datatype.builtin in instructions:
        ncs.add(instructions[datatype.builtin])
    else:
        raise CompileError(f"Cannot reserve storage for type '{datatype.builtin.name.lower()}'")
    return datatype.size(root)


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
    ):
        super().__init__()
        self.identifier: Identifier = identifier
        self.data_type: DynamicDataType = data_type
        self.expression: Expression = value

    def register(self, root: CodeRoot) -> None:
        root.register_global(
            self.identifier,
            self.data_type,
            origin=self.require_source_origin(),
        )

    def compile(self, ncs: NCS, root: CodeRoot):
        # Only emitted global slots participate in stack-offset lookup.
        declaration = GlobalVariableDeclaration(self.identifier, self.data_type)
        declaration._emit_storage(ncs, root)

        block = CodeBlock(
            CompilationContext(
                SemanticContext(
                    source_origin=self.require_source_origin(),
                    global_initializer=True,
                )
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
        stack_index = scoped.offset - scoped.datatype.size(root)
        ncs.instructions.append(
            NCSInstruction(
                NCSInstructionType.CPDOWNSP,
                [stack_index, scoped.datatype.size(root)],
            ),
        )
        initializer_size = scoped.datatype.size(root)
        ncs.add(NCSInstructionType.MOVSP, args=[-initializer_size])
        block.context.stack.consume(initializer_size)


class GlobalVariableDeclaration(TopLevelObject):
    def __init__(self, identifier: Identifier, data_type: DynamicDataType):
        super().__init__()
        self.identifier: Identifier = identifier
        self.data_type: DynamicDataType = data_type

    def register(self, root: CodeRoot) -> None:
        root.register_global(
            self.identifier,
            self.data_type,
            origin=self.require_source_origin(),
        )

    def compile(self, ncs: NCS, root: CodeRoot):  # noqa: A003
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

        root.allocate_global(self.identifier, self.data_type)


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
        *,
        global_initializer: bool = False,
    ):
        self.function_name = function_name
        self.global_initializer = global_initializer
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


class SourceIdentifierKind(Enum):
    """Identifier classes for predefined constants and known functions."""

    FUNCTION = "function"
    CONSTANT = "constant"


@dataclass(frozen=True)
class FunctionSignature:
    """The parts of a function declaration that define its type signature."""

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

    @classmethod
    def from_engine_function(cls, function: ScriptFunction) -> FunctionSignature:
        """Build the source signature of an engine function."""
        return cls(
            return_type=DynamicDataType(function.returntype),
            parameter_types=tuple(
                DynamicDataType(parameter.datatype) for parameter in function.params
            ),
        )


@dataclass(frozen=True)
class EngineFunctionReference:
    """One predefined engine routine and its ACTION opcode routine index."""

    routine_id: int
    function: ScriptFunction


@dataclass
class FunctionSymbol:
    """A function signature, declaration sites, and optional implementation entry."""

    name: str
    signature: FunctionSignature
    parameters: tuple[FunctionDefinitionParam, ...]
    implementation: FunctionDefinition | None = None
    entry_instruction: NCSInstruction | None = None
    called_functions: set[str] = field(default_factory=set)
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


class GetScopedResult(NamedTuple):
    is_global: bool
    datatype: DynamicDataType
    offset: int


class Struct:
    def __init__(self, identifier: Identifier, members: list[StructMember]):
        self.identifier: Identifier = identifier
        self.members: list[StructMember] = members
        self._cached_size: int | None = None

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
    """Compilation state, global storage, and top-level symbols."""

    def __init__(
        self,
        constants: list[ScriptConstant],
        functions: list[ScriptFunction],
        library_lookup: list[str] | list[Path] | list[Path | str] | str | Path | None,
        library: dict[str, bytes],
        max_include_depth: int = DEFAULT_MAX_INCLUDE_DEPTH,
        *,
        source_encoding: str | None = None,
    ):
        self.objects: list[TopLevelObject] = []
        self.source_encoding = source_encoding

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
        self._predefined_constants: dict[str, ConstantValue] = {}
        for constant in constants:
            if constant.datatype == DataType.INT:
                value: int | float | str = _int32(int(constant.value))
            elif constant.datatype == DataType.FLOAT:
                value = canonicalize_kotor_float_constant(float(constant.value))
            elif constant.datatype == DataType.STRING:
                value = str(constant.value)
            else:
                continue
            self._predefined_constants[constant.name] = ConstantValue(constant.datatype, value)
        self.struct_map: dict[str, Struct] = {}
        self._struct_origins: dict[str, SourceOrigin] = {}
        # Registered names and allocated global slots are tracked separately.
        self._registered_globals: dict[str, ScopedValue] = {}
        self._global_origins: dict[str, SourceOrigin] = {}
        self._global_function_references: set[str] = set()
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
                continue

            origin = SourceOrigin(file_level, self._next_source_order)
            self._next_source_order += 1
            obj.set_source_origin(origin)
            expanded.append(obj)
        return expanded

    def register_symbols(self) -> None:
        """Collect struct layouts, then register declarations in source order."""
        if self._symbols_registered:
            return

        struct_definitions = [obj for obj in self.objects if isinstance(obj, StructDefinition)]
        for definition in struct_definitions:
            definition.register(self)
        self._validate_struct_definitions()

        for obj in self.objects:
            if isinstance(obj, StructDefinition):
                self._validate_struct_source_identifiers(obj)
            elif isinstance(obj, (GlobalVariableDeclaration, GlobalVariableInitialization)):
                obj.register(self)
            elif isinstance(obj, (FunctionForwardDeclaration, FunctionDefinition)):
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
        self._struct_origins[name] = definition.require_source_origin()

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

    def is_function_identifier(
        self,
        name: str,
        source_origin: SourceOrigin | None,
    ) -> bool:
        """Check whether a function name is visible at this source position."""
        return (
            self.get_engine_function(name) is not None
            or self.get_visible_function(name, source_origin) is not None
        )

    def classify_source_identifier(
        self,
        name: str,
        source_origin: SourceOrigin | None,
    ) -> SourceIdentifierKind | None:
        """Classify a visible constant or function; variables and struct names are unclassified."""
        if self.get_compile_time_constant(name) is not None:
            return SourceIdentifierKind.CONSTANT
        if self.is_function_identifier(name, source_origin):
            return SourceIdentifierKind.FUNCTION
        return None

    def require_variable_identifier(
        self,
        identifier: Identifier,
        source_origin: SourceOrigin,
        *,
        context: str,
    ) -> None:
        """Reject names already classified as functions or constants."""
        kind = self.classify_source_identifier(identifier.label, source_origin)
        if kind is None:
            return
        raise CompileError(
            f"Identifier '{identifier.label}' cannot be used as {context}\n"
            f"  It is already classified as a {kind.value} identifier"
        )

    def validate_data_type(
        self,
        datatype: DynamicDataType,
        *,
        context: str,
        allow_void: bool,
        source_origin: SourceOrigin | None = None,
    ) -> None:
        if datatype.builtin == DataType.VOID:
            if not allow_void:
                raise CompileError(f"Invalid void type for {context}")
            return
        if datatype.builtin == DataType.STRUCT:
            struct_name = datatype._struct  # noqa: SLF001
            if not struct_name or struct_name not in self.struct_map:
                raise CompileError(f"Unknown struct type '{struct_name}' for {context}")
            if (
                source_origin is not None
                and self.classify_source_identifier(struct_name, source_origin) is not None
            ):
                raise CompileError(
                    f"Struct type '{struct_name}' is not available for {context}\n"
                    "  The name is already classified as a function or constant identifier"
                )

    def require_struct_layout_at(
        self,
        datatype: DynamicDataType,
        origin: SourceOrigin,
        *,
        context: str,
    ) -> None:
        """Require an earlier struct layout for global storage and by-value members."""
        if datatype.builtin != DataType.STRUCT:
            return
        definition = self._struct_origins.get(datatype.struct_name)
        if definition is None or definition.source_order >= origin.source_order:
            raise CompileError(
                f"Struct type '{datatype.struct_name}' is not yet defined for {context}"
            )

    def _validate_struct_source_identifiers(self, definition: StructDefinition) -> None:
        """Validate struct/member tokens at the point this struct appears in source."""
        origin = definition.require_source_origin()
        struct_name = definition.identifier.label
        self.require_variable_identifier(
            definition.identifier,
            origin,
            context="a struct name",
        )
        for member in definition.members:
            self.require_variable_identifier(
                member.identifier,
                origin,
                context=f"a member name in struct '{struct_name}'",
            )
            self.validate_data_type(
                member.datatype,
                context=f"member '{member.identifier.label}' of struct '{struct_name}'",
                allow_void=False,
                source_origin=origin,
            )
            self.require_struct_layout_at(
                member.datatype,
                origin,
                context=f"member '{member.identifier.label}' of struct '{struct_name}'",
            )

    def register_global(
        self,
        identifier: Identifier,
        datatype: DynamicDataType,
        *,
        origin: SourceOrigin,
    ) -> None:
        """Register a runtime global without allocating its VM storage yet."""
        name = identifier.label
        self.require_variable_identifier(
            identifier,
            origin,
            context="a global variable name",
        )
        self.validate_data_type(
            datatype,
            context=f"global variable '{name}'",
            allow_void=False,
            source_origin=origin,
        )

        self.require_struct_layout_at(datatype, origin, context=f"global variable '{name}'")
        if name in self._registered_globals:
            raise CompileError(f"Identifier '{identifier}' is already declared")
        self._registered_globals[name] = ScopedValue(identifier, datatype)
        self._global_origins[name] = origin

    def register_function(
        self,
        function: FunctionForwardDeclaration | FunctionDefinition,
    ) -> None:
        """Register a function at its declaration position."""
        name = function.identifier.label
        origin = function.require_source_origin()
        is_implementation = isinstance(function, FunctionDefinition)

        if self.get_compile_time_constant(name) is not None:
            raise CompileError(
                f"Function '{name}' conflicts with a compile-time constant identifier"
            )

        self.validate_data_type(
            function.return_type,
            context=f"return type of function '{name}'",
            allow_void=True,
            source_origin=origin,
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

            # Parameters see earlier declarations, not the function name being registered.
            self.require_variable_identifier(
                parameter.identifier,
                origin,
                context=f"a parameter name in function '{name}'",
            )
            self.validate_data_type(
                parameter.data_type,
                context=f"parameter '{parameter_name}' of function '{name}'",
                allow_void=False,
                source_origin=origin,
            )

        _validate_and_normalize_default_parameters(function.parameters, self, name)
        signature = FunctionSignature.from_function(function)

        engine = self.get_engine_function(name)
        if engine is not None:
            engine_signature = FunctionSignature.from_engine_function(engine.function)
            if signature != engine_signature:
                self._raise_function_signature_mismatch(
                    name,
                    engine_signature,
                    signature,
                )
            if is_implementation:
                raise CompileError(
                    f"Function '{name}' is already implemented by the engine\n"
                    "  A predefined engine function may only be redeclared as an identical prototype"
                )
            # Matching API prototypes remain ACTION calls.
            return

        existing = self.function_map.get(name)
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

    def note_function_reference(self, callee: FunctionSymbol, caller: str | None) -> None:
        """Record a call from its owning function, or from a global initializer if None."""
        if caller is None:
            self._global_function_references.add(callee.name)
        else:
            self.function_map[caller].called_functions.add(callee.name)

    def validate_referenced_functions(self, entry: FunctionSymbol) -> set[str]:
        """Require implementations only in the entry/global-initializer call graph."""
        pending = [entry.name, *sorted(self._global_function_references)]
        required: set[str] = set()
        while pending:
            name = pending.pop()
            if name in required:
                continue
            required.add(name)
            pending.extend(sorted(self.function_map[name].called_functions))

        unresolved = [
            name for name in required if self.function_map[name].implementation is None
        ]
        if not unresolved:
            return required

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

    def validate_semantics(self) -> None:
        """Validate all initializers and bodies, including unused functions."""
        from pykotor.resource.formats.ncs.compiler.semantic import (  # noqa: PLC0415
            ExpressionSemanticAnalyzer,
            FunctionSemanticAnalyzer,
            LocalSemanticScope,
        )

        for obj in self.objects:
            if isinstance(obj, GlobalVariableInitialization):
                analyzer = ExpressionSemanticAnalyzer(
                    self, obj.require_source_origin(), caller=None, global_initializer=True
                )
                analyzer.validate_initializer(
                    obj.expression, obj.data_type, obj.identifier.label, LocalSemanticScope()
                )
            elif isinstance(obj, FunctionDefinition):
                FunctionSemanticAnalyzer(self, obj).validate()

    def compile(self, ncs: NCS):  # noqa: A003
        self.expand_includes()

        self.register_symbols()
        self.validate_semantics()

        main_symbol = self.function_map.get("main")
        conditional_symbol = self.function_map.get("StartingConditional")
        if main_symbol is not None and main_symbol.has_root_declaration:
            entry_symbol = main_symbol
        elif conditional_symbol is not None and conditional_symbol.has_root_declaration:
            entry_symbol = conditional_symbol
        else:
            raise EntryPointError(
                "This file has no entry point and cannot be compiled (Most likely an include file)."
            )
        self._validate_entry_point(entry_symbol)
        entry_instruction = self._require_entry_implementation(entry_symbol)
        is_conditional = entry_symbol.name == "StartingConditional"

        required_functions = self.validate_referenced_functions(entry_symbol)

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
        implementations = [
            obj for obj in self.objects
            if isinstance(obj, FunctionDefinition) and obj.identifier.label in required_functions
        ]

        global_code = NCS()
        for global_def in script_globals:
            global_def.compile(global_code, self)
        function_code = NCS()
        for obj in implementations:
            obj.compile(function_code, self)

        # Keep the script result below globals so it survives cleanup.
        if is_conditional:
            ncs.add(NCSInstructionType.RSADDI)
        ncs.merge(global_code)

        global_bytes = -self.scope_size()
        if global_bytes:
            # SAVEBP occupies one four-byte stack cell.
            ncs.add(NCSInstructionType.SAVEBP)
            if is_conditional:
                ncs.add(NCSInstructionType.RSADDI)

        ncs.add(NCSInstructionType.JSR, jump=entry_instruction)

        if global_bytes:
            if is_conditional:
                # Stack: [script result][globals][saved BP][function result].
                ncs.add(NCSInstructionType.CPDOWNSP, args=[-(global_bytes + 12), 4])
                ncs.add(NCSInstructionType.MOVSP, args=[-4])
            ncs.add(NCSInstructionType.RESTOREBP)
            ncs.add(NCSInstructionType.MOVSP, args=[-global_bytes])
        ncs.add(NCSInstructionType.RETN)

        ncs.merge(function_code)

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
        start_instruction = symbol.require_entry_instruction()

        if len(args_list) > len(parameters):
            raise CompileError(
                f"Too many arguments in call to '{name}'\n"
                f"  Expected at most: {len(parameters)}\n"
                f"  Got: {len(args_list)}"
            )

        required_params = [param for param in parameters if param.default is None]

        if len(required_params) > len(args_list):
            required_names = [p.identifier.label for p in required_params]
            msg = (
                f"Missing required parameters in call to '{name}'\n"
                f"  Required: {', '.join(required_names)}\n"
                f"  Provided {len(args_list)} of {len(parameters)} parameters"
            )
            raise CompileError(msg)

        while len(parameters) > len(args_list):
            param_index = len(args_list)
            default_expr = parameters[param_index].default
            if default_expr is None:
                msg = f"Missing default value for parameter {param_index} in '{name}'"
                raise CompileError(msg)
            args_list.append(default_expr)

        return_type_size = emit_value_reservation(ncs, self, return_type)

        # Include the return slot when computing argument offsets.
        block.context.stack.push(return_type_size)

        offset = 0

        # Push arguments right-to-left, keeping each paired with its parameter.
        call_arguments = list(zip(parameters, args_list))
        for param, arg in reversed(call_arguments):
            arg_datatype: DynamicDataType = arg.compile(ncs, self, block)
            offset += arg_datatype.size(self)
            if param.data_type != arg_datatype:
                msg = (
                    f"Parameter type mismatch in call to '{name}'\n"
                    f"  Parameter '{param.identifier}' expects: {param.data_type.builtin.name}\n"
                    f"  Got: {arg_datatype.builtin.name}"
                )
                raise CompileError(msg)
        # JSR consumes arguments and leaves the return slot.
        block.context.stack.consume(offset)
        ncs.add(NCSInstructionType.JSR, jump=start_instruction)

        return return_type

    def get_compile_time_constant(
        self,
        identifier: Identifier | str,
    ) -> ConstantValue | None:
        """Look up a predefined constant in the identifier specification."""
        label = identifier.label if isinstance(identifier, Identifier) else identifier
        return self._predefined_constants.get(label)

    def get_registered_global(
        self,
        identifier: Identifier | str,
        *,
        source_origin: SourceOrigin | None = None,
    ) -> ScopedValue | None:
        """Resolve a global; initializers may only see slots allocated by that point."""
        label = identifier.label if isinstance(identifier, Identifier) else identifier
        origin = self._global_origins.get(label)
        if source_origin is not None and origin is not None:
            if origin.source_order > source_origin.source_order:
                return None
        return self._registered_globals.get(label)

    def registered_global_names(self) -> tuple[str, ...]:
        """Return registered runtime-global names for source-level diagnostics."""
        return tuple(self._registered_globals)

    def allocate_global(
        self,
        identifier: Identifier,
        datatype: DynamicDataType,
    ) -> None:
        """Record one VM global slot after its allocation instructions are emitted."""
        registered = self._registered_globals.get(identifier.label)
        if registered is None:
            raise ValueError(
                f"Internal compiler error: global '{identifier}' was emitted before registration"
            )
        if registered.data_type != datatype:
            raise ValueError(
                f"Internal compiler error: emitted global '{identifier}' does not match registration"
            )
        if any(scoped.identifier == identifier for scoped in self._global_scope):
            raise ValueError(
                f"Internal compiler error: global '{identifier}' storage was emitted twice"
            )
        self._global_scope.insert(0, ScopedValue(identifier, datatype))

    def get_scoped(self, identifier: Identifier, root: CodeRoot) -> GetScopedResult:
        offset = 0
        for scoped in self._global_scope:
            offset -= scoped.data_type.size(root)
            if scoped.identifier == identifier:
                break
        else:
            available = [s.identifier.label for s in self._global_scope[:10]]
            more = len(self._global_scope) - 10
            more_text = f" (and {more} more)" if more > 0 else ""
            msg = f"Undefined variable '{identifier}'\n  Available globals: {', '.join(available)}{more_text}"
            raise CompileError(msg)
        return GetScopedResult(
            is_global=True, datatype=scoped.data_type, offset=offset
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

        for index, statement in enumerate(self._statements):
            statement.compile(
                ncs,
                root,
                self,
                return_instruction,
                break_instruction,
                continue_instruction,
            )
            if isinstance(statement, ReturnStatement):
                # A later case label can still be reached after a return.
                remaining = self._statements[index + 1 :]
                if not any(statement_contains_switch_label(item) for item in remaining):
                    self.context.stack.restore(entry_transient_bytes)
                    return

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
        self, identifier: Identifier, data_type: DynamicDataType
    ):
        # Previously retained temporaries sit below this new local.
        value = ScopedValue(
            identifier,
            data_type,
            temporary_bytes_below=self.context.stack.temporary_bytes,
        )
        self.scope.insert(0, value)

    def get_scoped(
        self,
        identifier: Identifier,
        root: CodeRoot,
        offset: int | None = None,
    ) -> GetScopedResult:
        # Subtract only temporaries above the local, once across parent scopes.
        offset = -self.context.stack.temporary_bytes if offset is None else offset
        for scoped in self.scope:
            offset -= scoped.data_type.size(root)
            if scoped.identifier == identifier:
                break
        else:
            if self._parent is not None:
                return self._parent.get_scoped(identifier, root, offset)
            scoped = root.get_scoped(identifier, root)
            if self.context.semantic.global_initializer:
                return GetScopedResult(
                    is_global=False,
                    datatype=scoped.datatype,
                    offset=scoped.offset + offset,
                )
            return scoped
        return GetScopedResult(
            is_global=False,
            datatype=scoped.data_type,
            offset=offset + scoped.temporary_bytes_below,
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
        """Match conservative non-void return-path analysis."""
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
    def __init__(
        self,
        identifier: Identifier,
        data_type: DynamicDataType,
        *,
        temporary_bytes_below: int = 0,
    ):
        self.identifier: Identifier = identifier
        self.data_type: DynamicDataType = data_type
        # Temporary depth at allocation; zero for parameters and globals.
        self.temporary_bytes_below: int = temporary_bytes_below


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
        return


class FunctionDefinition(TopLevelObject):
    """A function signature and its implementation body."""

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

        # Parameter zero is nearest SP; add_scoped() inserts at the front.
        for param in reversed(parameters):
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


def resolve_kotor_constant_expression(
    expression: Expression,
    root: CodeRoot,
    *,
    allow_numeric_negation: bool = False,
) -> ConstantValue | None:
    """Resolve literals and predefined constants, preserving type restrictions.

    Numeric negation is allowed only when allow_numeric_negation is set.
    """
    while isinstance(expression, ParenthesizedExpression):
        expression = expression.expression

    if isinstance(expression, IntExpression):
        return ConstantValue(DataType.INT, expression.value)
    if isinstance(expression, FloatExpression):
        return ConstantValue(DataType.FLOAT, expression.value)
    if isinstance(expression, StringExpression):
        return ConstantValue(DataType.STRING, expression.value)
    if isinstance(expression, ObjectExpression):
        return ConstantValue(DataType.OBJECT, expression.value)
    if isinstance(expression, IdentifierExpression):
        return root.get_compile_time_constant(expression.identifier)

    if allow_numeric_negation and isinstance(expression, UnaryOperatorExpression):
        operand = resolve_kotor_constant_expression(
            expression.expression1,
            root,
            allow_numeric_negation=False,
        )
        if operand is None:
            return None

        instructions = {mapping.instruction for mapping in expression.compatibility}
        if operand.datatype == DataType.INT and NCSInstructionType.NEGI in instructions:
            return ConstantValue(DataType.INT, _int32(-int(operand.value)))
        if operand.datatype == DataType.FLOAT and NCSInstructionType.NEGF in instructions:
            return ConstantValue(DataType.FLOAT, _float32(-float(operand.value)))

    return None


def _normalize_default_parameter(
    parameter: FunctionDefinitionParam,
    root: CodeRoot,
    function_name: str,
) -> None:
    """Validate and normalize an optional parameter value."""
    expression = parameter.default
    if expression is None:
        return

    datatype = parameter.data_type.builtin
    parameter_name = parameter.identifier.label

    if datatype in (DataType.INT, DataType.FLOAT, DataType.STRING):
        constant = resolve_kotor_constant_expression(
            expression,
            root,
            allow_numeric_negation=False,
        )
        if constant is None:
            raise CompileError(
                f"Non-constant default value for parameter '{parameter_name}' in '{function_name}'"
                "\n  KotOR defaults must be literals or predefined constants"
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

    if datatype == DataType.OBJECT:
        constant = resolve_kotor_constant_expression(
            expression, root, allow_numeric_negation=False
        )
        if constant is None or constant.datatype != DataType.OBJECT:
            raise CompileError(
                f"Non-constant default value for parameter '{parameter_name}' in '{function_name}'"
                "\n  Object defaults must be OBJECT_SELF or OBJECT_INVALID"
            )
        parameter.default = ObjectExpression(int(constant.value))
        return

    if datatype == DataType.VECTOR:
        while isinstance(expression, ParenthesizedExpression):
            expression = expression.expression
        if not isinstance(expression, VectorExpression):
            raise CompileError(
                f"Non-constant default value for parameter '{parameter_name}' in '{function_name}'"
                "\n  Vector defaults must be vector literals"
            )

        parameter.default = VectorExpression(
            *(FloatExpression(value) for value in expression.resolve_components(root))
        )
        return

    raise CompileError(
        f"Type '{datatype.name.lower()}' does not support default parameters"
        f"\n  Parameter: {parameter_name} in '{function_name}'"
    )


def _validate_and_normalize_default_parameters(
    parameters: list[FunctionDefinitionParam],
    root: CodeRoot,
    function_name: str,
) -> None:
    """Validate parameter defaults, their types, and required-before-optional order."""
    optional_parameters_started = False
    for parameter in parameters:
        if parameter.default is None:
            if optional_parameters_started:
                raise CompileError(
                    "Function parameter without a default value can't follow one with a default value."
                )
            continue

        optional_parameters_started = True
        _normalize_default_parameter(
            parameter,
            root,
            function_name,
        )


class IncludeScript(TopLevelObject):
    def __init__(
        self,
        file: StringExpression,
        library: dict[str, bytes] | None = None,
        *,
        location: SourceLocation | None = None,
    ):
        self.location = location
        self.file: StringExpression = file
        self.library: dict[str, bytes] = {} if library is None else library

    def expand(
        self,
        root: CodeRoot,
        context: IncludeContext,
    ) -> list[TopLevelObject]:
        """Parse this include and recursively expand its nested includes."""
        try:
            canonical_name = context.begin_include(self.file.value)
        except CompileError as exc:
            exc.add_include(self.location)
            raise
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
        except CompileError as exc:
            exc.add_include(self.location)
            raise
        finally:
            context.end_include(canonical_name, completed=completed)

    def _parse(self, root: CodeRoot) -> CodeRoot:
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
            source_encoding=root.source_encoding,
        )
        source_name, source_bytes = self._get_script(root)
        return parser.parse(source_bytes, source_name=source_name)

    def compile(self, ncs: NCS, root: CodeRoot):  # noqa: A003
        raise RuntimeError(
            "Internal compiler error: include directive reached bytecode emission"
        )

    def _get_script(self, root: CodeRoot) -> tuple[str, bytes]:
        """Load bytes and their identity; the parser uses the shared source decoder."""
        for folder in root.library_lookup:
            filepath = folder / f"{self.file.value}.nss"
            if filepath.is_file():
                try:
                    return str(filepath), filepath.read_bytes()
                except OSError as exc:
                    raise MissingIncludeError(
                        f"Failed to read include file '{filepath}': {exc}"
                    ) from exc

        canonical_name = IncludeContext.canonicalize(self.file.value)
        for library_name, source_bytes in self.library.items():
            if IncludeContext.canonicalize(library_name) == canonical_name:
                return f"{library_name}.nss", source_bytes

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
        return None


class Expression(ABC):
    """Base class for expressions that leave a value on the stack."""

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
    ) -> DynamicDataType:
        """Compile an expression and record its result in the shared stack context."""
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
    """Base class for executable statements."""

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
    """An identifier-rooted path to assignable storage."""

    @classmethod
    def from_expression(cls, expression: Expression) -> FieldAccess | None:
        """Unwrap parentheses/members only; calls and other values are not lvalues."""
        members: list[Identifier] = []
        while True:
            if isinstance(expression, ParenthesizedExpression):
                expression = expression.expression
            elif isinstance(expression, MemberAccessExpression):
                members.append(expression.member)
                expression = expression.base
            elif isinstance(expression, IdentifierExpression):
                return cls([expression.identifier, *reversed(members)])
            elif isinstance(expression, FieldAccessExpression):
                return cls([*expression.field_access.identifiers, *reversed(members)])
            else:
                return None

    @classmethod
    def require_lvalue(cls, expression: Expression) -> FieldAccess:
        access = cls.from_expression(expression)
        if access is None:
            raise CompileError("Assignment/increment target must be a variable or a member of a variable")
        return access

    def __init__(self, identifiers: list[Identifier]):
        super().__init__()
        self.identifiers: list[Identifier] = identifiers

    def get_scoped(self, block: CodeBlock, root: CodeRoot) -> GetScopedResult:
        """Resolve storage and member offsets for this field path."""
        if len(self.identifiers) == 0:
            msg = "Internal error: FieldAccess has no identifiers"
            raise CompileError(msg)

        first_ident: Identifier = self.identifiers[0]
        source_origin = block.context.semantic.source_origin
        if root.get_compile_time_constant(first_ident) is not None:
            raise CompileError(f"Cannot assign to compile-time constant '{first_ident}'")
        if root.is_function_identifier(first_ident.label, source_origin):
            raise CompileError(
                f"Function identifier '{first_ident.label}' cannot be used as a variable"
            )
        scoped: GetScopedResult = block.get_scoped(first_ident, root)

        is_global: bool = scoped.is_global
        offset: int = scoped.offset
        datatype: DynamicDataType = scoped.datatype

        for member in self.identifiers[1:]:
            layout = resolve_member_layout(root, datatype, member)
            offset += layout.offset
            datatype = layout.datatype

        return GetScopedResult(is_global, datatype, offset)

    def compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        is_global, variable_type, stack_index = self.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))
        return variable_type


class MemberLayout(NamedTuple):
    """Type and byte offset of one field within an aggregate value."""

    datatype: DynamicDataType
    offset: int


def resolve_member_layout(
    root: CodeRoot, datatype: DynamicDataType, member: Identifier
) -> MemberLayout:
    """Use the same member names/layout for semantic checks, reads and writes."""
    if datatype.builtin == DataType.VECTOR:
        offsets = {"x": 0, "y": 4, "z": 8}
        if member.label in offsets:
            return MemberLayout(DynamicDataType.FLOAT, offsets[member.label])
    elif datatype.builtin == DataType.STRUCT:
        struct_type = root.struct_map.get(datatype.struct_name)
        if struct_type is None:
            raise CompileError(f"Unknown struct type '{datatype.struct_name}' in member access")
        return MemberLayout(
            struct_type.child_type(root, member), struct_type.child_offset(root, member)
        )
    raise CompileError(
        f"Attempting to access unknown member '{member.label}' on datatype '{datatype}'"
    )


class MemberAccessExpression(Expression):
    """Read a field from a variable or an aggregate-valued expression."""

    def __init__(self, base: Expression, member: Identifier):
        super().__init__()
        self.base = base
        self.member = member
        self.resolved_type: DynamicDataType | None = None

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        storage = FieldAccess.from_expression(self)
        if storage is not None:
            return storage.compile(ncs, root, block)

        member_type = self.resolved_type
        if member_type is None:
            raise ValueError("Internal compiler error: member expression was not semantically validated")
        # Keep the typed result below the temporary aggregate.
        member_size = emit_value_reservation(ncs, root, member_type)
        block.context.stack.push(member_size)

        base_type = self.base.compile(ncs, root, block)
        member = resolve_member_layout(root, base_type, self.member)
        if member.datatype != member_type:
            raise ValueError("Internal compiler error: member type changed after semantic analysis")
        base_size = base_type.size(root)
        ncs.add(NCSInstructionType.CPTOPSP, args=[member.offset - base_size, member_size])
        block.context.stack.push(member_size)
        ncs.add(NCSInstructionType.CPDOWNSP, args=[-(base_size + 2 * member_size), member_size])
        ncs.add(NCSInstructionType.MOVSP, args=[-(base_size + member_size)])
        block.context.stack.consume(base_size + member_size)
        return member_type


class ParenthesizedExpression(Expression):
    """Preserve source parentheses where grammar distinctions depend on them."""

    def __init__(self, expression: Expression):
        super().__init__()
        self.expression = expression

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        return self.expression.compile(ncs, root, block)


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

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        source_origin = block.context.semantic.source_origin
        constant = root.get_compile_time_constant(self.identifier)
        if constant is not None:
            return _emit_constant(ncs, constant)
        if root.is_function_identifier(self.identifier.label, source_origin):
            raise CompileError(
                f"Function identifier '{self.identifier.label}' cannot be used as a value"
            )

        is_global, datatype, stack_index = block.get_scoped(self.identifier, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, datatype.size(root)])
        return datatype

    def get_constant(self, root: CodeRoot) -> ScriptConstant | None:
        return next(
            (constant for constant in root.constants if constant.name == self.identifier.label),
            None,
        )

    def is_constant(self, root: CodeRoot) -> bool:
        return root.get_compile_time_constant(self.identifier) is not None


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

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        ncs.instructions.append(
            NCSInstruction(NCSInstructionType.CONSTI, [_int32(self.value)])
        )
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

    @classmethod
    def from_engine_default(cls, value: int | str) -> ObjectExpression:
        """Convert numeric or named object defaults to CONSTO operands."""
        if value == "OBJECT_SELF":
            return cls(0)
        if value == "OBJECT_INVALID":
            return cls(1)
        try:
            return cls(int(value))
        except (TypeError, ValueError) as exc:
            raise CompileError(f"Invalid engine object default: {value!r}") from exc

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

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        ncs.instructions.append(
            NCSInstruction(NCSInstructionType.CONSTF, [_float32(self.value)])
        )
        return DynamicDataType.FLOAT


class VectorExpression(Expression):
    def __init__(self, x: Expression, y: Expression, z: Expression):
        super().__init__()
        self.x: Expression = x
        self.y: Expression = y
        self.z: Expression = z

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

    def resolve_components(self, root: CodeRoot) -> tuple[float, float, float]:
        """Resolve the three literal/predefined float tokens, never runtime values."""
        values: list[float] = []
        for component in (self.x, self.y, self.z):
            constant = resolve_kotor_constant_expression(component, root)
            if constant is None or constant.datatype != DataType.FLOAT:
                raise CompileError(
                    "Vector components must be float literals or predefined float constants"
                )
            values.append(float(constant.value))
        return values[0], values[1], values[2]

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        values = self.resolve_components(root)
        for value in values:
            FloatExpression(value).compile(ncs, root, block)
        return DynamicDataType.VECTOR


class EngineCallExpression(Expression):
    """Engine call node for directly constructed ASTs."""

    def __init__(
        self,
        function: ScriptFunction,
        routine_id: int,
        args: list[Expression],
    ):
        super().__init__()
        self._function: ScriptFunction = function
        self._routine_id: int = routine_id
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
                    args.append(ObjectExpression.from_engine_default(param.default))
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
                args.append(ObjectExpression.from_engine_default(constant.value))

        ordinary_argument_bytes = 0

        for reverse_index, arg in enumerate(reversed(args)):
            param_index = len(args) - 1 - reverse_index
            param = self._function.params[param_index]
            param_type = DynamicDataType(param.datatype)
            if param_type == DataType.ACTION:
                after_command = NCSInstruction()
                ncs.add(
                    NCSInstructionType.STORE_STATE,
                    args=[-root.scope_size(), block.stack_depth(root)],
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
            return root.compile_jsr(ncs, block, symbol, *self._args)

        engine = root.get_engine_function(name)
        if engine is not None:
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

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
        and_mapping = next(
            (
                mapping
                for mapping in self.compatibility
                if mapping.instruction == NCSInstructionType.LOGANDII
            ),
            None,
        )
        if and_mapping is not None:
            return self._compile_short_circuit_and(ncs, root, block, and_mapping)

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
        block.context.stack.replace(type1_size + type2_size, result_size)
        return result_type

    def _compile_short_circuit_and(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        mapping: BinaryOperatorMapping,
    ) -> DynamicDataType:
        """Skip the right operand of false AND and leave a normalized integer result."""
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

        ncs.add(NCSInstructionType.CPTOPSP, args=[-lhs_size, lhs_size])
        block.context.stack.push(lhs_size)
        ncs.add(NCSInstructionType.JZ, jump=end_label)
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

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
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

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:  # noqa: A003
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


class Assignment(Expression):
    def __init__(
        self,
        field_access: FieldAccess,
        value: Expression,
    ):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        variable_type = self.expression.compile(ncs, root, block)

        is_global, expression_type, stack_index = self.field_access.get_scoped(
            block,
            root,
        )

        instruction_type = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP

        if variable_type != expression_type:
            var_name = ".".join(str(ident) for ident in self.field_access.identifiers)
            msg = f"Type mismatch in assignment to '{var_name}'\n  Variable type: {expression_type.builtin.name}\n  Expression type: {variable_type.builtin.name}"
            raise CompileError(msg)

        ncs.instructions.append(
            NCSInstruction(instruction_type, [stack_index, expression_type.size(root)]),
        )


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
        is_global, variable_type, stack_index = self.field_access.get_scoped(
            block,
            root,
        )
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expresion_type = self.expression.compile(ncs, root, block)

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

        ncs.add(arthimetic_instruction, args=[])

        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class SubtractionAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        isglobal, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if isglobal else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expresion_type = self.expression.compile(ncs, root, block)

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

        ncs.add(arthimetic_instruction)

        ins_cpdown = NCSInstructionType.CPDOWNBP if isglobal else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if isglobal else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class MultiplicationAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        isglobal, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if isglobal else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expresion_type = self.expression.compile(ncs, root, block)

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

        ncs.add(arthimetic_instruction)

        ins_cpdown = NCSInstructionType.CPDOWNBP if isglobal else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if isglobal else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class DivisionAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        isglobal, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if isglobal else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expresion_type = self.expression.compile(ncs, root, block)

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

        ncs.add(arthimetic_instruction)

        ins_cpdown = NCSInstructionType.CPDOWNBP if isglobal else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if isglobal else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class ModuloAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        isglobal, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if isglobal else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expresion_type = self.expression.compile(ncs, root, block)

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

        ncs.add(arthimetic_instruction)

        ins_cpdown = NCSInstructionType.CPDOWNBP if isglobal else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if isglobal else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expresion_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class BitwiseAndAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        is_global, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expression_type = self.expression.compile(ncs, root, block)

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

        ncs.add(bitwise_instruction, args=[])

        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class BitwiseOrAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        is_global, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expression_type = self.expression.compile(ncs, root, block)

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

        ncs.add(bitwise_instruction, args=[])

        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class BitwiseXorAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        is_global, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expression_type = self.expression.compile(ncs, root, block)

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

        ncs.add(bitwise_instruction, args=[])

        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class BitwiseLeftAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        is_global, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expression_type = self.expression.compile(ncs, root, block)

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

        ncs.add(bitwise_instruction, args=[])

        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class BitwiseRightAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        is_global, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expression_type = self.expression.compile(ncs, root, block)

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

        ncs.add(bitwise_instruction, args=[])

        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        return variable_type


class BitwiseUnsignedRightAssignment(Expression):
    def __init__(self, field_access: FieldAccess, value: Expression):
        super().__init__()
        self.field_access: FieldAccess = field_access
        self.expression: Expression = value

    def _compile(self, ncs: NCS, root: CodeRoot, block: CodeBlock) -> DynamicDataType:
        is_global, variable_type, stack_index = self.field_access.get_scoped(block, root)
        instruction_type = NCSInstructionType.CPTOPBP if is_global else NCSInstructionType.CPTOPSP
        ncs.add(instruction_type, args=[stack_index, variable_type.size(root)])
        block.context.stack.push(variable_type.size(root))

        expression_type = self.expression.compile(ncs, root, block)

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

        ncs.add(bitwise_instruction, args=[])

        ins_cpdown = NCSInstructionType.CPDOWNBP if is_global else NCSInstructionType.CPDOWNSP
        offset_cpdown = stack_index if is_global else stack_index - variable_type.size(root)
        ncs.add(ins_cpdown, args=[offset_cpdown, variable_type.size(root)])

        block.context.stack.replace(
            variable_type.size(root) + expression_type.size(root),
            variable_type.size(root),
        )
        return variable_type


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
    ):
        super().__init__()
        self.data_type: DynamicDataType = data_type
        self.declarators: list[VariableDeclarator | VariableInitializer] = declarators

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        for declarator in self.declarators:
            declarator.compile(ncs, root, block, self.data_type)


class VariableDeclarator:
    def __init__(self, identifier: Identifier):
        self.identifier: Identifier = identifier

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        data_type: DynamicDataType,
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

        block.add_scoped(self.identifier, data_type)


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
    ):
        declarator = VariableDeclarator(self.identifier)
        declarator.compile(ncs, root, block, data_type)

        # Discard the assignment result after storing the initializer.
        assignment = Assignment(
            FieldAccess([self.identifier]),
            self.expression,
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
        """Emit the conditional branches and their bodies."""
        end_label = NCSInstruction(NCSInstructionType.NOP, args=[])
        needs_end_label = False

        for condition_and_block in self.if_blocks:
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
        # Restore tracked depth for other paths reaching later labels.
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

        isglobal, variable_type, stack_index = self.field_access.get_scoped(block, root)
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

        isglobal, variable_type, stack_index = self.field_access.get_scoped(block, root)
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

        isglobal, variable_type, stack_index = self.field_access.get_scoped(block, root)
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

        isglobal, variable_type, stack_index = self.field_access.get_scoped(block, root)
        if isglobal:
            ncs.add(NCSInstructionType.DECIBP, args=[stack_index])
        else:
            ncs.add(NCSInstructionType.DECISP, args=[stack_index])

        return variable_type


class SwitchStatement(Statement):
    def __init__(self, expression: Expression, body: CodeBlock):
        super().__init__()
        self.expression = expression
        self.body = body

    def _validate_labels(
        self,
        root: CodeRoot,
        source_origin: SourceOrigin | None,
    ) -> tuple[list[tuple[ExpressionSwitchLabel, int]], DefaultSwitchLabel | None]:
        """Resolve integer case values and check label uniqueness."""
        cases: list[tuple[ExpressionSwitchLabel, int]] = []
        case_values: set[int] = set()
        default_label: DefaultSwitchLabel | None = None

        for label in iter_switch_labels(self.body):
            if isinstance(label, DefaultSwitchLabel):
                if default_label is not None:
                    raise ValueError(
                        "Internal compiler error: duplicate default label reached emission"
                    )
                default_label = label
                continue

            constant = resolve_kotor_constant_expression(
                label.expression,
                root,
                allow_numeric_negation=True,
            )
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
            cases.append((label, value))

        return cases, default_label

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ):
        cases, default_label = self._validate_labels(
            root,
            block.context.semantic.source_origin,
        )

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

        # Emit the body once; dispatch targets its case/default labels.
        tempncs = NCS()
        block.context.control.push(target)
        try:
            self.body.compile(
                tempncs,
                root,
                block,
                return_instruction,
                end_of_switch,
                continue_instruction,
            )
        finally:
            block.context.control.pop(target)

        for label, case_value in cases:
            ncs.add(NCSInstructionType.CPTOPSP, args=[-4, 4])
            ncs.add(NCSInstructionType.CONSTI, args=[case_value])
            ncs.add(NCSInstructionType.EQUALII, args=[])
            ncs.add(NCSInstructionType.JNZ, jump=label.target)

        # Check every case before selecting default.
        if default_label is not None:
            ncs.add(NCSInstructionType.JMP, jump=default_label.target)
        else:
            ncs.add(NCSInstructionType.JMP, jump=end_of_switch)

        ncs.merge(tempncs)
        ncs.instructions.append(end_of_switch)

        # Only the discriminant remains at this exit.
        ncs.add(NCSInstructionType.MOVSP, args=[-expression_type.size(root)])
        block.context.stack.consume(expression_type.size(root))


class SwitchLabel(Statement):
    """Statement-level switch label with a stable dispatcher jump target."""

    def __init__(self):
        super().__init__()
        self.target = NCSInstruction(NCSInstructionType.NOP, args=[])

    def compile(
        self,
        ncs: NCS,
        root: CodeRoot,
        block: CodeBlock,
        return_instruction: NCSInstruction,
        break_instruction: NCSInstruction | None,
        continue_instruction: NCSInstruction | None,
    ) -> None:
        ncs.instructions.append(self.target)


class ExpressionSwitchLabel(SwitchLabel):
    def __init__(self, expression: Expression):
        super().__init__()
        self.expression = expression


class DefaultSwitchLabel(SwitchLabel):
    pass


def iter_switch_labels(block: CodeBlock) -> Iterator[SwitchLabel]:
    """Yield labels in this switch, excluding nested switches."""
    pending: list[Statement | CodeBlock] = list(reversed(block.statements))
    while pending:
        statement = pending.pop()
        if isinstance(statement, SwitchLabel):
            yield statement
            continue
        if isinstance(statement, SwitchStatement):
            # Nested switches own their labels.
            continue
        if isinstance(statement, CodeBlock):
            pending.extend(reversed(statement.statements))
            continue
        if isinstance(statement, ScopedBlock):
            pending.extend(reversed(statement.block.statements))
            continue
        if isinstance(statement, ConditionalBlock):
            child_blocks = [branch.block for branch in statement.if_blocks]
            if statement.else_block is not None:
                child_blocks.append(statement.else_block)
            for child in reversed(child_blocks):
                pending.extend(reversed(child.statements))
            continue
        if isinstance(statement, (WhileLoopBlock, DoWhileLoopBlock, ForLoopBlock)):
            pending.extend(reversed(statement.block.statements))


def statement_contains_switch_label(statement: Statement | CodeBlock) -> bool:
    """Return whether an outer switch may dispatch into this statement subtree."""
    if isinstance(statement, SwitchLabel):
        return True
    if isinstance(statement, SwitchStatement):
        return False
    if isinstance(statement, CodeBlock):
        return any(statement_contains_switch_label(child) for child in statement.statements)
    if isinstance(statement, ScopedBlock):
        return statement_contains_switch_label(statement.block)
    if isinstance(statement, ConditionalBlock):
        if any(statement_contains_switch_label(branch.block) for branch in statement.if_blocks):
            return True
        return (
            statement.else_block is not None
            and statement_contains_switch_label(statement.else_block)
        )
    if isinstance(statement, (WhileLoopBlock, DoWhileLoopBlock, ForLoopBlock)):
        return statement_contains_switch_label(statement.block)
    return False


def _statement_returns_on_all_paths(statement: Statement | CodeBlock) -> bool:
    """Check whether every statically known path returns."""
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
