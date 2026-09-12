"""Local semantic analysis for NSS function bodies.

This module deliberately contains no NCS emission logic.  It validates source-level
names, types, lexical scopes, and control-flow legality before the emitter allocates
local stack slots or writes instructions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Protocol, cast

from pykotor.common.script import DataType
from pykotor.resource.formats.ncs import NCSInstructionType
from pykotor.resource.formats.ncs.compiler.classes import (
    AdditionAssignment,
    Assignment,
    BinaryOperatorExpression,
    BitwiseAndAssignment,
    BitwiseLeftAssignment,
    BitwiseNotExpression,
    BitwiseOrAssignment,
    BitwiseRightAssignment,
    BitwiseUnsignedRightAssignment,
    BitwiseXorAssignment,
    BreakStatement,
    CodeBlock,
    CompileError,
    ConditionalBlock,
    ContinueStatement,
    DeclarationStatement,
    DefaultSwitchLabel,
    DivisionAssignment,
    DoWhileLoopBlock,
    DynamicDataType,
    EmptyStatement,
    EngineCallExpression,
    Expression,
    ExpressionStatement,
    ExpressionSwitchLabel,
    FieldAccess,
    FieldAccessExpression,
    FloatExpression,
    ForLoopBlock,
    FunctionCallExpression,
    FunctionDefinition,
    FunctionSymbol,
    IdentifierExpression,
    IntExpression,
    LogicalNotExpression,
    ModuloAssignment,
    MultiplicationAssignment,
    NopStatement,
    ObjectExpression,
    PostfixDecrementExpression,
    PostfixIncrementExpression,
    PrefixDecrementExpression,
    PrefixIncrementExpression,
    ReturnStatement,
    ScopedBlock,
    StringExpression,
    SubtractionAssignment,
    SwitchStatement,
    TernaryConditionalExpression,
    UnaryOperatorExpression,
    VariableDeclarator,
    VariableInitializer,
    VectorExpression,
    WhileLoopBlock,
)

if TYPE_CHECKING:
    from pykotor.common.script import ScriptFunction
    from pykotor.resource.formats.ncs.compiler.classes import CodeRoot, Identifier


def _normalize_int32(value: int) -> int:
    """Normalize case values to the signed 32-bit representation used by NCS."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


@dataclass(frozen=True)
class LocalSemanticSymbol:
    """One source-level local or parameter visible during semantic analysis."""

    name: str
    data_type: DynamicDataType
    is_const: bool = False


class LocalSemanticScope:
    """A lexical source scope independent of VM stack layout."""

    def __init__(self, parent: LocalSemanticScope | None = None):
        self.parent = parent
        self._symbols: dict[str, LocalSemanticSymbol] = {}

    def declare(
        self,
        identifier: Identifier,
        data_type: DynamicDataType,
        *,
        is_const: bool = False,
    ) -> LocalSemanticSymbol:
        name = identifier.label
        if name in self._symbols:
            raise CompileError(
                f"Variable '{name}' is already declared in this scope"
                "\n  A variable name may only be declared once per lexical scope"
            )
        symbol = LocalSemanticSymbol(name, data_type, is_const)
        self._symbols[name] = symbol
        return symbol

    def resolve(self, name: str) -> LocalSemanticSymbol | None:
        scope: LocalSemanticScope | None = self
        while scope is not None:
            symbol = scope._symbols.get(name)
            if symbol is not None:
                return symbol
            scope = scope.parent
        return None


@dataclass(frozen=True)
class ResolvedValue:
    """Type and mutability of a resolved local/global field access."""

    data_type: DynamicDataType
    is_const: bool


class CompoundAssignmentExpression(Protocol):
    """Shared source shape of all compound-assignment AST nodes."""

    field_access: FieldAccess
    expression: Expression


@dataclass(frozen=True)
class CompoundAssignmentRule:
    """Allowed operand pairs for one compound-assignment AST node."""

    operand_pairs: tuple[tuple[DynamicDataType, DynamicDataType], ...]
    description: str


class FunctionSemanticAnalyzer:
    """Validate one function body before any local bytecode is emitted."""

    _LOCAL_TYPES = {
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

    _COMPOUND_ASSIGNMENTS: dict[type[Expression], CompoundAssignmentRule] = {
        AdditionAssignment: CompoundAssignmentRule(
            (
                (DynamicDataType.INT, DynamicDataType.INT),
                (DynamicDataType.FLOAT, DynamicDataType.FLOAT),
                (DynamicDataType.FLOAT, DynamicDataType.INT),
                (DynamicDataType.STRING, DynamicDataType.STRING),
                (DynamicDataType.VECTOR, DynamicDataType.VECTOR),
            ),
            "int+=int, float+=float/int, string+=string, vector+=vector",
        ),
        SubtractionAssignment: CompoundAssignmentRule(
            (
                (DynamicDataType.INT, DynamicDataType.INT),
                (DynamicDataType.FLOAT, DynamicDataType.FLOAT),
                (DynamicDataType.FLOAT, DynamicDataType.INT),
                (DynamicDataType.VECTOR, DynamicDataType.VECTOR),
            ),
            "int-=int, float-=float/int, vector-=vector",
        ),
        MultiplicationAssignment: CompoundAssignmentRule(
            (
                (DynamicDataType.INT, DynamicDataType.INT),
                (DynamicDataType.FLOAT, DynamicDataType.FLOAT),
                (DynamicDataType.FLOAT, DynamicDataType.INT),
                (DynamicDataType.VECTOR, DynamicDataType.FLOAT),
            ),
            "int*=int, float*=float/int, vector*=float",
        ),
        DivisionAssignment: CompoundAssignmentRule(
            (
                (DynamicDataType.INT, DynamicDataType.INT),
                (DynamicDataType.FLOAT, DynamicDataType.FLOAT),
                (DynamicDataType.FLOAT, DynamicDataType.INT),
                (DynamicDataType.VECTOR, DynamicDataType.FLOAT),
            ),
            "int/=int, float/=float/int, vector/=float",
        ),
        ModuloAssignment: CompoundAssignmentRule(
            ((DynamicDataType.INT, DynamicDataType.INT),),
            "int%=int",
        ),
        BitwiseAndAssignment: CompoundAssignmentRule(
            ((DynamicDataType.INT, DynamicDataType.INT),),
            "int&=int",
        ),
        BitwiseOrAssignment: CompoundAssignmentRule(
            ((DynamicDataType.INT, DynamicDataType.INT),),
            "int|=int",
        ),
        BitwiseXorAssignment: CompoundAssignmentRule(
            ((DynamicDataType.INT, DynamicDataType.INT),),
            "int^=int",
        ),
        BitwiseLeftAssignment: CompoundAssignmentRule(
            ((DynamicDataType.INT, DynamicDataType.INT),),
            "int<<=int",
        ),
        BitwiseRightAssignment: CompoundAssignmentRule(
            ((DynamicDataType.INT, DynamicDataType.INT),),
            "int>>=int",
        ),
        BitwiseUnsignedRightAssignment: CompoundAssignmentRule(
            ((DynamicDataType.INT, DynamicDataType.INT),),
            "int>>>=int",
        ),
    }

    def __init__(self, root: CodeRoot, function: FunctionDefinition):
        self.root = root
        self.function = function
        self.source_origin = function.require_source_origin()
        self._breakable_depth = 0
        self._loop_depth = 0

    def validate(self) -> None:
        """Validate parameters, every statement, and conservative return coverage."""
        function_scope = LocalSemanticScope()
        for parameter in self.function.parameters:
            function_scope.declare(parameter.identifier, parameter.data_type)

        # BioWare tracks the function parameter list and the outer compound body at
        # adjacent recursion levels. Locals in the body may therefore shadow a
        # parameter, while duplicate parameter names remain invalid during top-level
        # symbol registration.
        self._validate_block(self.function.block, function_scope, create_scope=True)

        if (
            self.function.return_type != DynamicDataType.VOID
            and not self.function.block.returns_on_all_paths()
        ):
            raise CompileError(
                "Not all control paths in function "
                f"'{self.function.identifier.label}' return a value"
                "\n  Non-void functions must return a value on every control path"
            )

    def _validate_block(
        self,
        block: CodeBlock,
        parent_scope: LocalSemanticScope,
        *,
        create_scope: bool = True,
    ) -> None:
        scope = LocalSemanticScope(parent_scope) if create_scope else parent_scope
        # Deliberately do not stop after return/break/continue. Semantic validity is
        # independent of whether emission later drops unreachable instructions.
        for statement in block.statements:
            self._validate_statement(statement, scope)

    def _validate_statement(
        self,
        statement: object,
        scope: LocalSemanticScope,
    ) -> None:
        if isinstance(statement, CodeBlock):
            self._validate_block(statement, scope)
            return
        if isinstance(statement, (EmptyStatement, NopStatement)):
            return
        if isinstance(statement, ExpressionStatement):
            self._expression_type(statement.expression, scope)
            return
        if isinstance(statement, DeclarationStatement):
            self._validate_declaration(statement, scope)
            return
        if isinstance(statement, ConditionalBlock):
            self._validate_conditional(statement, scope)
            return
        if isinstance(statement, ReturnStatement):
            self._validate_return(statement, scope)
            return
        if isinstance(statement, WhileLoopBlock):
            self._require_int_condition(statement.condition, scope, "while loop")
            with self._control_scope(loop=True):
                self._validate_block(statement.block, scope)
            return
        if isinstance(statement, DoWhileLoopBlock):
            with self._control_scope(loop=True):
                self._validate_block(statement.block, scope)
            self._require_int_condition(statement.condition, scope, "do-while loop")
            return
        if isinstance(statement, ForLoopBlock):
            self._validate_for(statement, scope)
            return
        if isinstance(statement, ScopedBlock):
            self._validate_block(statement.block, scope)
            return
        if isinstance(statement, BreakStatement):
            if self._breakable_depth == 0:
                raise CompileError(
                    "break statement not inside loop or switch"
                    "\n  break can only be used inside while, do-while, for, or switch statements"
                )
            return
        if isinstance(statement, ContinueStatement):
            if self._loop_depth == 0:
                raise CompileError(
                    "continue statement not inside loop"
                    "\n  continue can only be used inside while, do-while, or for loops"
                )
            return
        if isinstance(statement, SwitchStatement):
            self._validate_switch(statement, scope)
            return
        raise ValueError(
            f"Internal compiler error: no semantic validator for {type(statement).__name__}"
        )

    def _validate_declaration(
        self,
        declaration: DeclarationStatement,
        scope: LocalSemanticScope,
    ) -> None:
        if declaration.is_const:
            raise CompileError(
                "const keyword cannot be used on non-global variables"
                "\n  Declare compile-time constants at global scope"
            )
        self._validate_local_type(declaration.data_type)

        for declarator in declaration.declarators:
            scope.declare(declarator.identifier, declaration.data_type)
            if isinstance(declarator, VariableInitializer):
                initializer_type = self._expression_type(declarator.expression, scope)
                if initializer_type != declaration.data_type:
                    raise CompileError(
                        f"Type mismatch in initializer for '{declarator.identifier.label}'\n"
                        f"  Variable type: {self._type_name(declaration.data_type)}\n"
                        f"  Initializer type: {self._type_name(initializer_type)}"
                    )
            elif not isinstance(declarator, VariableDeclarator):
                raise ValueError(
                    "Internal compiler error: unknown local declarator "
                    f"{type(declarator).__name__}"
                )

    def _validate_local_type(self, data_type: DynamicDataType) -> None:
        self.root.validate_data_type(
            data_type,
            context="local variable",
            allow_void=False,
        )
        if data_type.builtin not in self._LOCAL_TYPES:
            raise CompileError(
                f"Unsupported local variable type '{data_type.builtin.name.lower()}'"
            )

    def _validate_conditional(
        self,
        conditional: ConditionalBlock,
        scope: LocalSemanticScope,
    ) -> None:
        for branch in conditional.if_blocks:
            self._require_int_condition(branch.condition, scope, "conditional expression")
            self._validate_block(branch.block, scope)
        if conditional.else_block is not None:
            self._validate_block(conditional.else_block, scope)

    def _validate_return(
        self,
        statement: ReturnStatement,
        scope: LocalSemanticScope,
    ) -> None:
        expected = self.function.return_type
        function_name = self.function.identifier.label
        if statement.expression is None:
            if expected != DynamicDataType.VOID:
                raise CompileError(
                    f"Function '{function_name}' must return {self._type_name(expected)}"
                )
            return

        if expected == DynamicDataType.VOID:
            raise CompileError(f"Void function '{function_name}' cannot return a value")

        actual = self._expression_type(statement.expression, scope)
        if actual != expected:
            raise CompileError(
                f"Return type mismatch in '{function_name}'\n"
                f"  Expected: {self._type_name(expected)}\n"
                f"  Got: {self._type_name(actual)}"
            )

    def _validate_for(self, statement: ForLoopBlock, scope: LocalSemanticScope) -> None:
        # The parser's BioWare-compatible grammar only permits an expression here,
        # but the AST keeps Statement support for callers constructing trees directly.
        if statement.initial is not None:
            if isinstance(statement.initial, Expression):
                self._expression_type(statement.initial, scope)
            else:
                self._validate_statement(statement.initial, scope)
        self._require_int_condition(statement.condition, scope, "for loop")
        if statement.iteration is not None:
            self._expression_type(statement.iteration, scope)
        with self._control_scope(loop=True):
            self._validate_block(statement.block, scope)

    def _validate_switch(
        self,
        statement: SwitchStatement,
        parent_scope: LocalSemanticScope,
    ) -> None:
        expression_type = self._expression_type(statement.expression, parent_scope)
        if expression_type != DynamicDataType.INT:
            raise CompileError(
                "Switch expression must evaluate to int; "
                f"got {self._type_name(expression_type)}"
            )

        seen_cases: set[int] = set()
        saw_default = False
        switch_scope = LocalSemanticScope(parent_scope)
        switch_scope_has_locals = False

        with self._control_scope(loop=False):
            for switch_block in statement.blocks:
                if switch_block.labels and switch_scope_has_locals:
                    raise CompileError(
                        "Switch case/default label jumps over local declarations"
                        "\n  Case labels must have the same stack depth as switch entry"
                    )

                for label in switch_block.labels:
                    if isinstance(label, DefaultSwitchLabel):
                        if saw_default:
                            raise CompileError(
                                "Switch statement contains more than one default label"
                            )
                        saw_default = True
                        continue
                    if not isinstance(label, ExpressionSwitchLabel):
                        raise ValueError(
                            "Internal compiler error: unknown switch label "
                            f"{type(label).__name__}"
                        )
                    constant = label.expression.constant_value(
                        self.root,
                        self.source_origin,
                    )
                    if constant is None or constant.datatype != DataType.INT:
                        raise CompileError(
                            "Switch case label must be a constant integer expression"
                        )
                    value = _normalize_int32(int(constant.value))
                    if value in seen_cases:
                        raise CompileError(f"Duplicate switch case value: {value}")
                    seen_cases.add(value)

                for child in switch_block.block:
                    self._validate_statement(child, switch_scope)
                    if isinstance(child, DeclarationStatement):
                        switch_scope_has_locals = True

    @contextmanager
    def _control_scope(self, *, loop: bool) -> Iterator[None]:
        self._breakable_depth += 1
        if loop:
            self._loop_depth += 1
        try:
            yield
        finally:
            if loop:
                self._loop_depth -= 1
            self._breakable_depth -= 1

    def _require_int_condition(
        self,
        expression: Expression,
        scope: LocalSemanticScope,
        description: str,
    ) -> None:
        data_type = self._expression_type(expression, scope)
        if data_type != DynamicDataType.INT:
            raise CompileError(
                f"{description.capitalize()} must be integer type, got {self._type_name(data_type)}"
                "\n  Conditions must evaluate to int (0 = false, non-zero = true)"
            )

    def _expression_type(
        self,
        expression: Expression,
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        if isinstance(expression, IntExpression):
            return DynamicDataType.INT
        if isinstance(expression, FloatExpression):
            return DynamicDataType.FLOAT
        if isinstance(expression, StringExpression):
            return DynamicDataType.STRING
        if isinstance(expression, ObjectExpression):
            return DynamicDataType.OBJECT
        if isinstance(expression, VectorExpression):
            return DynamicDataType.VECTOR
        if isinstance(expression, IdentifierExpression):
            constant = self.root.get_compile_time_constant(
                expression.identifier,
                self.source_origin,
            )
            if constant is not None:
                return DynamicDataType(constant.datatype)
            return self._resolve_identifier(expression.identifier.label, scope).data_type
        if isinstance(expression, FieldAccessExpression):
            return self._resolve_field_access(expression.field_access, scope).data_type
        if isinstance(expression, EngineCallExpression):
            return self._engine_call_expression_type(expression, scope)
        if isinstance(expression, FunctionCallExpression):
            return self._call_type(expression, scope)
        if isinstance(expression, BinaryOperatorExpression):
            return self._binary_type(expression, scope)
        if isinstance(expression, TernaryConditionalExpression):
            return self._ternary_type(expression, scope)
        if isinstance(expression, UnaryOperatorExpression):
            return self._unary_type(expression, scope)
        if isinstance(expression, LogicalNotExpression):
            operand_type = self._expression_type(expression.expression1, scope)
            if operand_type != DynamicDataType.INT:
                raise CompileError(
                    "Logical NOT requires integer operand, got "
                    f"{self._type_name(operand_type)}"
                )
            return DynamicDataType.INT
        if isinstance(expression, BitwiseNotExpression):
            operand_type = self._expression_type(expression.expression1, scope)
            if operand_type != DynamicDataType.INT:
                raise CompileError(
                    "Bitwise NOT (~) requires integer operand, got "
                    f"{self._type_name(operand_type)}"
                )
            return DynamicDataType.INT
        if isinstance(expression, Assignment):
            return self._assignment_type(expression, scope)
        if type(expression) in self._COMPOUND_ASSIGNMENTS:
            return self._compound_assignment_type(expression, scope)
        if isinstance(
            expression,
            (
                PrefixIncrementExpression,
                PostfixIncrementExpression,
                PrefixDecrementExpression,
                PostfixDecrementExpression,
            ),
        ):
            return self._increment_type(expression, scope)
        raise ValueError(
            f"Internal compiler error: no semantic type rule for {type(expression).__name__}"
        )

    def _resolve_identifier(
        self,
        name: str,
        scope: LocalSemanticScope,
    ) -> ResolvedValue:
        local = scope.resolve(name)
        if local is not None:
            return ResolvedValue(local.data_type, local.is_const)
        global_symbol = self.root.get_registered_global(name, self.source_origin)
        if global_symbol is not None:
            return ResolvedValue(global_symbol.data_type, global_symbol.is_const)

        available = sorted(self.root.registered_global_names(self.source_origin))[:10]
        suffix = f"\n  Available globals: {', '.join(available)}" if available else ""
        raise CompileError(f"Undefined variable '{name}'{suffix}")

    def _resolve_field_access(
        self,
        field_access: FieldAccess,
        scope: LocalSemanticScope,
        *,
        writable: bool = False,
        allow_const: bool = False,
    ) -> ResolvedValue:
        if not field_access.identifiers:
            raise ValueError("Internal compiler error: FieldAccess has no identifiers")

        first = field_access.identifiers[0]
        if self.root.get_compile_time_constant(first, self.source_origin) is not None:
            raise CompileError(f"Cannot assign to compile-time constant '{first}'")

        resolved = self._resolve_identifier(first.label, scope)
        data_type = resolved.data_type
        for member in field_access.identifiers[1:]:
            if data_type == DynamicDataType.VECTOR:
                if member.label not in {"x", "y", "z"}:
                    raise CompileError(
                        f"Attempting to access unknown member '{member.label}' on datatype 'vector'"
                    )
                data_type = DynamicDataType.FLOAT
                continue
            if data_type.builtin == DataType.STRUCT:
                struct_name = data_type.struct_name
                if struct_name is None or struct_name not in self.root.struct_map:
                    raise CompileError("Unknown struct type in field access")
                data_type = self.root.struct_map[struct_name].child_type(self.root, member)
                continue
            raise CompileError(
                f"Attempting to access unknown member '{member.label}' on datatype "
                f"'{self._type_name(data_type)}'"
            )

        if writable and resolved.is_const and not allow_const:
            name = ".".join(identifier.label for identifier in field_access.identifiers)
            raise CompileError(f"Cannot assign to const variable '{name}'")
        return ResolvedValue(data_type, resolved.is_const)

    def _engine_call_type(
        self,
        function: ScriptFunction,
        args: tuple[Expression, ...],
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        if len(args) > len(function.params):
            raise CompileError(
                f"Too many arguments for '{function.name}'\n"
                f"  Expected: {len(function.params)}, Got: {len(args)}"
            )

        required = [parameter for parameter in function.params if parameter.default is None]
        if len(args) < len(required):
            raise CompileError(
                f"Missing required arguments for '{function.name}'\n"
                f"  Required parameters: {', '.join(parameter.name for parameter in required)}\n"
                f"  Provided: {len(args)} argument(s)"
            )

        for index, argument in enumerate(args):
            actual = self._expression_type(argument, scope)
            parameter = function.params[index]
            if parameter.datatype == DataType.ACTION:
                if actual != DynamicDataType.VOID:
                    raise CompileError(
                        f"ACTION parameter '{parameter.name}' in call to "
                        f"'{function.name}' must be a void expression\n"
                        f"  Got: {self._type_name(actual)}"
                    )
                continue

            expected = DynamicDataType(parameter.datatype)
            if actual != expected:
                raise CompileError(
                    f"Type mismatch for parameter '{parameter.name}' in call to "
                    f"'{function.name}'\n"
                    f"  Expected: {self._type_name(expected)}\n"
                    f"  Got: {self._type_name(actual)}"
                )
        return DynamicDataType(function.returntype)

    def _call_type(
        self,
        expression: FunctionCallExpression,
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        name = expression.function_identifier.label
        symbol = self.root.get_visible_function(name, self.source_origin)
        if symbol is not None:
            return self._user_function_call_type(symbol, expression.arguments, scope)

        engine = self.root.get_engine_function(name)
        if engine is not None:
            return self._engine_call_type(engine.function, expression.arguments, scope)

        available = self.root.callable_names(self.source_origin)
        preview = available[:10]
        suffix = "..." if len(available) > 10 else ""
        raise CompileError(
            f"Undefined function '{name}'\n"
            f"  Available functions: {', '.join(preview)}{suffix}"
        )

    def _user_function_call_type(
        self,
        symbol: FunctionSymbol,
        args: tuple[Expression, ...],
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        name = symbol.name
        parameters = symbol.parameters
        if len(args) > len(parameters):
            raise CompileError(
                f"Too many arguments in call to '{name}'\n"
                f"  Expected at most: {len(parameters)}\n"
                f"  Got: {len(args)}"
            )
        required_count = sum(parameter.default is None for parameter in parameters)
        if len(args) < required_count:
            raise CompileError(
                f"Missing required parameters in call to '{name}'\n"
                f"  Provided {len(args)} of {len(parameters)} parameters"
            )

        symbol.note_reference()
        for parameter, argument in zip(parameters, args):
            actual = self._expression_type(argument, scope)
            if actual != parameter.data_type:
                raise CompileError(
                    f"Parameter type mismatch in call to '{name}'\n"
                    f"  Parameter '{parameter.identifier.label}' expects: "
                    f"{self._type_name(parameter.data_type)}\n"
                    f"  Got: {self._type_name(actual)}"
                )
        return symbol.signature.return_type

    def _engine_call_expression_type(
        self,
        expression: EngineCallExpression,
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        """Validate an explicitly constructed engine-call AST node."""
        return self._engine_call_type(expression.function, expression.arguments, scope)

    def _binary_type(
        self,
        expression: BinaryOperatorExpression,
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        left = self._expression_type(expression.expression1, scope)
        right = self._expression_type(expression.expression2, scope)
        for mapping in expression.compatibility:
            if left == mapping.lhs and right == mapping.rhs:
                return DynamicDataType(mapping.result)

        if left == right and left.builtin in {DataType.STRUCT, DataType.VECTOR}:
            instructions = {mapping.instruction for mapping in expression.compatibility}
            equality_instructions = {
                NCSInstructionType.EQUALII,
                NCSInstructionType.EQUALFF,
                NCSInstructionType.EQUALOO,
                NCSInstructionType.EQUALSS,
                NCSInstructionType.EQUALEFFEFF,
                NCSInstructionType.EQUALEVTEVT,
                NCSInstructionType.EQUALLOCLOC,
                NCSInstructionType.EQUALTALTAL,
                NCSInstructionType.NEQUALII,
                NCSInstructionType.NEQUALFF,
                NCSInstructionType.NEQUALOO,
                NCSInstructionType.NEQUALSS,
                NCSInstructionType.NEQUALEFFEFF,
                NCSInstructionType.NEQUALEVTEVT,
                NCSInstructionType.NEQUALLOCLOC,
                NCSInstructionType.NEQUALTALTAL,
            }
            if instructions & equality_instructions:
                return DynamicDataType.INT

        raise CompileError(
            "Incompatible types for binary operation: "
            f"{self._type_name(left)} and {self._type_name(right)}"
        )

    def _ternary_type(
        self,
        expression: TernaryConditionalExpression,
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        condition = self._expression_type(expression.condition, scope)
        if condition != DynamicDataType.INT:
            raise CompileError(
                f"Ternary condition must be integer type, got {self._type_name(condition)}"
            )
        true_type = self._expression_type(expression.true_expr, scope)
        false_type = self._expression_type(expression.false_expr, scope)
        if true_type != false_type:
            raise CompileError(
                "Type mismatch in ternary operator\n"
                f"  True branch type: {self._type_name(true_type)}\n"
                f"  False branch type: {self._type_name(false_type)}"
            )
        return true_type

    def _unary_type(
        self,
        expression: UnaryOperatorExpression,
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        operand = self._expression_type(expression.expression1, scope)
        if any(operand == mapping.rhs for mapping in expression.compatibility):
            return operand
        supported = ", ".join(mapping.rhs.name.lower() for mapping in expression.compatibility)
        raise CompileError(
            f"Incompatible type for unary operation: {self._type_name(operand)}\n"
            f"  Supported types: {supported}"
        )

    def _assignment_type(
        self,
        expression: Assignment,
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        destination = self._resolve_field_access(
            expression.field_access,
            scope,
            writable=True,
            allow_const=expression.allow_const,
        )
        value_type = self._expression_type(expression.expression, scope)
        if destination.data_type != value_type:
            name = ".".join(identifier.label for identifier in expression.field_access.identifiers)
            raise CompileError(
                f"Type mismatch in assignment to '{name}'\n"
                f"  Variable type: {self._type_name(destination.data_type)}\n"
                f"  Expression type: {self._type_name(value_type)}"
            )
        return destination.data_type

    def _compound_assignment_type(
        self,
        expression: Expression,
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        compound = cast(CompoundAssignmentExpression, expression)
        field_access = compound.field_access
        destination = self._resolve_field_access(field_access, scope, writable=True)
        rhs_type = self._expression_type(compound.expression, scope)
        rule = self._COMPOUND_ASSIGNMENTS[type(expression)]
        if (destination.data_type, rhs_type) not in rule.operand_pairs:
            name = ".".join(identifier.label for identifier in field_access.identifiers)
            raise CompileError(
                f"Type mismatch in compound assignment on '{name}'\n"
                f"  Variable type: {self._type_name(destination.data_type)}\n"
                f"  Expression type: {self._type_name(rhs_type)}\n"
                f"  Supported: {rule.description}"
            )
        return destination.data_type

    def _increment_type(
        self,
        expression: PrefixIncrementExpression
        | PostfixIncrementExpression
        | PrefixDecrementExpression
        | PostfixDecrementExpression,
        scope: LocalSemanticScope,
    ) -> DynamicDataType:
        resolved = self._resolve_field_access(expression.field_access, scope, writable=True)
        if resolved.data_type != DynamicDataType.INT:
            name = ".".join(identifier.label for identifier in expression.field_access.identifiers)
            raise CompileError(
                f"Increment/decrement operator requires integer variable; '{name}' is "
                f"{self._type_name(resolved.data_type)}"
            )
        return DynamicDataType.INT

    @staticmethod
    def _type_name(data_type: DynamicDataType) -> str:
        if data_type.builtin == DataType.STRUCT:
            return f"struct {data_type.struct_name}"
        return data_type.builtin.name.lower()
