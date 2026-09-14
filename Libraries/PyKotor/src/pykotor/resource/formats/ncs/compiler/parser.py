"""PLY grammar and AST construction for NSS."""

from __future__ import annotations

from utility.system.path import Path
from typing import TYPE_CHECKING, NoReturn, Sequence, cast

from ply import yacc

from pykotor.resource.formats.ncs.compiler.classes import (
    AdditionAssignment,
    Assignment,
    BinaryOperatorExpression,
    BitwiseAndAssignment,
    BitwiseLeftAssignment,
    BitwiseOrAssignment,
    BitwiseRightAssignment,
    BitwiseUnsignedRightAssignment,
    BitwiseXorAssignment,
    BreakStatement,
    CodeBlock,
    CodeRoot,
    CompileError,
    ConditionAndBlock,
    ConditionalBlock,
    ContinueStatement,
    DeclarationStatement,
    DEFAULT_MAX_INCLUDE_DEPTH,
    DefaultSwitchLabel,
    DivisionAssignment,
    DoWhileLoopBlock,
    DynamicDataType,
    EmptyStatement,
    ExpressionStatement,
    ExpressionSwitchLabel,
    FieldAccess,
    FloatExpression,
    ForLoopBlock,
    FunctionCallExpression,
    FunctionDefinition,
    FunctionDefinitionParam,
    FunctionForwardDeclaration,
    GlobalVariableDeclaration,
    GlobalVariableInitialization,
    Identifier,
    IdentifierExpression,
    IncludeScript,
    IntExpression,
    MemberAccessExpression,
    ModuloAssignment,
    MultiplicationAssignment,
    ParenthesizedExpression,
    PostfixDecrementExpression,
    PostfixIncrementExpression,
    PrefixDecrementExpression,
    PrefixIncrementExpression,
    ReturnStatement,
    StructDefinition,
    StructMember,
    SubtractionAssignment,
    SwitchStatement,
    TernaryConditionalExpression,
    UnaryOperatorExpression,
    VariableDeclarator,
    VariableInitializer,
    VectorExpression,
    WhileLoopBlock,
)
from pykotor.resource.formats.ncs.compiler.lexer import NssLexer
from pykotor.resource.formats.ncs.compiler.source import SourceDocument, describe_token
from pykotor.resource.formats.ncs.string_encoding import encode_ncs_string

if TYPE_CHECKING:
    from pykotor.common.script import DataType, ScriptConstant, ScriptFunction
    from pykotor.resource.formats.ncs.compiler.classes import (
        Expression,
        TopLevelObject,
    )
else:
    from pykotor.common.script import DataType


class NssParser:
    """Parse NSS source into an abstract syntax tree."""

    def __init__(
        self,
        functions: list[ScriptFunction],
        constants: list[ScriptConstant],
        library: dict[str, bytes],
        library_lookup: Sequence[str | Path] | None = None,
        errorlog: yacc.NullLogger | None = yacc.NullLogger(),  # noqa: B008
        *,
        debug: bool = False,
        max_include_depth: int = DEFAULT_MAX_INCLUDE_DEPTH,
        source_encoding: str | None = None,
    ):
        self.source_encoding = source_encoding
        self.source: SourceDocument | None = None
        self.parser: yacc.LRParser = yacc.yacc(
            module=self,
            errorlog=errorlog,
            write_tables=False,
            debug=debug,
        )
        self.functions: list[ScriptFunction] = functions
        self.constants: list[ScriptConstant] = constants
        self.library: dict[str, bytes] = library
        self.max_include_depth = max_include_depth
        if library_lookup is None:
            lookup_items: list[str | Path] = []
        elif isinstance(library_lookup, (str, Path)):
            lookup_items = [library_lookup]
        else:
            lookup_items = list(library_lookup)
        self.library_lookup = [Path.pathify(item) for item in lookup_items]

    tokens: list[str] = NssLexer.tokens
    literals: list[str] = NssLexer.literals

    precedence: tuple[tuple[str, ...], ...] = (
        ("left", "OR"),
        ("left", "AND"),
        ("left", "BITWISE_OR"),
        ("left", "BITWISE_XOR"),
        ("left", "BITWISE_AND"),
        ("left", "EQUALS", "NOT_EQUALS"),
        ("left", "GREATER_THAN", "LESS_THAN", "GREATER_THAN_OR_EQUALS", "LESS_THAN_OR_EQUALS"),
        ("left", "BITWISE_LEFT", "BITWISE_RIGHT", "BITWISE_UNSIGNED_RIGHT"),
        ("left", "ADD", "MINUS"),
        ("left", "MULTIPLY", "DIVIDE", "MOD"),
    )

    def parse(self, source: str | bytes, *, source_name: str = "<string>", debug: bool = False) -> CodeRoot:
        """Parse one named source using an independent lexer and the shared decoder."""
        lexer = NssLexer(source_name=source_name, source_encoding=self.source_encoding)
        self.source = None
        return self.parser.parse(source, lexer=lexer.lexer, tracking=True, debug=debug)

    def p_error(self, p) -> NoReturn:
        if p is None:
            location = self.source.location(len(self.source.text)) if self.source is not None else None
            raise CompileError("unexpected end of file", location=location)
        raise CompileError(
            f"unexpected {describe_token(p.lexeme)}",
            location=p.location,
        )

    def _require_lvalue(self, production, index: int) -> FieldAccess:
        """Report an invalid storage target at its actual source expression."""
        try:
            return FieldAccess.require_lvalue(production[index])
        except CompileError as exc:
            location = self.source.location(production.lexpos(index)) if self.source is not None else None
            raise CompileError(exc.message, location=location) from None

    def p_code_root(self, p):
        """
        code_root : code_root code_root_object
                  |
        """  # noqa: D205, D415, D400, D212
        if len(p) == 3:
            code_root = cast("CodeRoot", p[1])
            parsed_object = p[2]
            if isinstance(parsed_object, list):
                code_root.objects.extend(parsed_object)
            else:
                code_root.objects.append(parsed_object)
            p[0] = code_root
        else:
            self.source = p.lexer.source
            p[0] = CodeRoot(
                constants=self.constants,
                functions=self.functions,
                library_lookup=self.library_lookup,
                library=self.library,
                max_include_depth=self.max_include_depth,
                source_encoding=self.source_encoding,
            )

    def p_code_root_object(self, p):
        """
        code_root_object : include_script
                         | typed_top_level_object
                         | struct_definition
        """  # noqa: D205, D415, D400, D212
        p[0] = p[1]

    def p_struct_definition(self, p):
        """
        struct_definition : STRUCT IDENTIFIER '{' struct_members '}' ';'
        """  # noqa: D415, D400, D212, D200
        p[0] = StructDefinition(p[2], p[4])

    def p_struct_members(self, p):
        """
        struct_members : struct_members struct_member
                       |
        """  # noqa: D415, D400, D212, D205
        if len(p) == 3:  # noqa: PLR2004
            cast("list", p[1]).extend(p[2])
            p[0] = p[1]
        else:
            p[0] = []

    def p_struct_member(self, p):
        """
        struct_member : non_void_data_type struct_member_declarators ';'
        """  # noqa: D200, D400, D212, D415
        p[0] = [StructMember(p[1], identifier) for identifier in p[2]]

    def p_struct_member_declarators(self, p):
        """
        struct_member_declarators : struct_member_declarators ',' IDENTIFIER
                                  | IDENTIFIER
        """  # noqa: D400, D212, D415, D205
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        else:
            p[0] = [p[1]]

    def p_include_script(self, p):
        """
        include_script : INCLUDE STRING_VALUE
        """  # noqa: D200, D400, D212, D415
        p[0] = IncludeScript(p[2], library=self.library, location=p.slice[1].location)

    @staticmethod
    def _global_objects(
        data_type: DynamicDataType,
        identifier: Identifier,
        tail: tuple[Expression | None, list[VariableDeclarator | VariableInitializer]],
    ) -> list[TopLevelObject]:
        initializer, trailing_declarators = tail
        declarators = [
            VariableInitializer(identifier, initializer)
            if initializer is not None
            else VariableDeclarator(identifier),
            *trailing_declarators,
        ]
        objects: list[TopLevelObject] = []
        for declarator in declarators:
            if isinstance(declarator, VariableInitializer):
                objects.append(
                    GlobalVariableInitialization(
                        declarator.identifier,
                        data_type,
                        declarator.expression,
                    )
                )
            else:
                objects.append(
                    GlobalVariableDeclaration(
                        declarator.identifier,
                        data_type,
                    )
                )
        return objects

    def p_typed_top_level_object(self, p):
        """
        typed_top_level_object : any_data_type IDENTIFIER '(' function_definition_params ')' ';'
                               | any_data_type IDENTIFIER '(' function_definition_params ')' '{' code_block '}'
                               | any_data_type IDENTIFIER global_variable_tail ';'
        """  # noqa: D400, D212, D415, D205
        data_type = p[1]
        identifier = p[2]
        if p[3] == '(':
            parameters = p[4]
            if len(p) == 7:
                p[0] = FunctionForwardDeclaration(data_type, identifier, parameters)
            else:
                block = p[7]
                p[0] = FunctionDefinition(data_type, identifier, parameters, block, p.lineno(2))
            return

        if data_type == DynamicDataType.VOID:
            raise CompileError(
                f"Cannot declare global variable '{identifier}' with void type\n"
                "  void can only be used as a function return type",
                location=p.slice[2].location,
            )
        p[0] = self._global_objects(data_type, identifier, p[3])

    def p_global_variable_tail(self, p):
        """
        global_variable_tail : '=' expression
                             | ',' variable_declarators
                             |
        """  # noqa: D400, D212, D415, D205
        if len(p) == 1:
            p[0] = (None, [])
        elif p[1] == "=":
            p[0] = (p[2], [])
        else:
            p[0] = (None, p[2])

    def p_function_definition_params(self, p):
        """
        function_definition_params : parameter_list
                                   | parameter_list ','
                                   |
        """  # noqa: D400, D212, D415, D205
        p[0] = p[1] if len(p) > 1 else []

    def p_parameter_list(self, p):
        """
        parameter_list : function_definition_param
                       | parameter_list ',' function_definition_param
        """  # noqa: D400, D212, D415, D205
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[1].append(p[3])
            p[0] = p[1]

    def p_function_definition_param(self, p):
        """
        function_definition_param : non_void_data_type IDENTIFIER
        """  # noqa: D200, D400, D212, D415
        p[0] = FunctionDefinitionParam(p[1], p[2])

    def p_function_definition_param_with_default(self, p):
        """
        function_definition_param : non_void_data_type IDENTIFIER '=' expression
        """  # noqa: D200, D400, D212, D415
        p[0] = FunctionDefinitionParam(p[1], p[2], p[4])

    def p_code_block(self, p):
        """
        code_block : code_block block_item
                   |
        """  # noqa: D400, D212, D415, D205
        if len(p) == 3:
            block: CodeBlock = p[1]
            block.add(p[2])
            p[0] = block
        elif len(p) == 1:
            p[0] = CodeBlock()

    def p_block_item(self, p):
        """
        block_item : statement
                   | declaration_statement
        """  # noqa: D400, D212, D415, D205
        p[0] = p[1]

    @staticmethod
    def _statement_block(statement) -> CodeBlock:
        if isinstance(statement, CodeBlock):
            return statement
        block = CodeBlock()
        block.add(statement)
        return block

    def p_while_loop(self, p):
        """
        while_loop : WHILE_CONTROL '(' expression ')' statement
        """  # noqa: D200, D400, D212, D415
        p[0] = WhileLoopBlock(p[3], self._statement_block(p[5]))

    def p_do_while_loop(self, p):
        """
        do_while_loop : DO_CONTROL statement WHILE_CONTROL '(' expression ')' ';'
        """  # noqa: D200, D400, D212, D415
        p[0] = DoWhileLoopBlock(p[5], self._statement_block(p[2]))

    def p_for_loop(self, p):
        """
        for_loop : FOR_CONTROL '(' optional_expression ';' optional_expression ';' optional_expression ')' statement
        """  # noqa: D200, D400, D212, D415
        initial = p[3]
        condition = p[5] if p[5] is not None else IntExpression(1)
        iteration = p[7]
        p[0] = ForLoopBlock(initial, condition, iteration, self._statement_block(p[9]))

    def p_optional_expression(self, p):
        """
        optional_expression : expression
                            |
        """  # noqa: D400, D212, D415, D205
        p[0] = p[1] if len(p) == 2 else None

    def p_scoped_block(self, p):
        """
        scoped_block : '{' code_block '}'
        """  # noqa: D200, D400, D212, D415
        p[0] = p[2]

    def p_statement(self, p):
        """
        statement : ';'
                  | non_null_statement
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = EmptyStatement() if p[1] == ";" else p[1]

    def p_non_null_statement(self, p):
        """
        non_null_statement : condition_statement
                           | return_statement
                           | while_loop
                           | do_while_loop
                           | for_loop
                           | switch_statement
                           | break_statement
                           | continue_statement
                           | scoped_block
                           | switch_label
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = p[1]

    def p_expression_statement(self, p):
        """
        non_null_statement : expression ';'
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = ExpressionStatement(p[1])

    def p_break_statement(self, p):
        """
        break_statement : BREAK_CONTROL ';'
        """  # noqa: D200, D400, D212, D415
        p[0] = BreakStatement()

    def p_continue_statement(self, p):
        """
        continue_statement : CONTINUE_CONTROL ';'
        """  # noqa: D200, D400, D212, D415
        p[0] = ContinueStatement()

    def p_declaration_statement(self, p):
        """
        declaration_statement : non_void_data_type variable_declarators ';'
        """  # noqa: D200, D400, D212, D415
        p[0] = DeclarationStatement(p[1], p[2])

    def p_variable_declarators(self, p):
        """
        variable_declarators : uninitialized_declarators
                             | uninitialized_declarators '=' expression
        """  # noqa: D400, D212, D415, D205
        declarators = p[1]
        if len(p) == 4:
            declarators[-1] = VariableInitializer(declarators[-1].identifier, p[3])
        p[0] = declarators

    def p_uninitialized_declarators(self, p):
        """
        uninitialized_declarators : IDENTIFIER
                                  | uninitialized_declarators ',' IDENTIFIER
        """  # noqa: D400, D212, D415, D205
        if len(p) == 2:
            p[0] = [VariableDeclarator(p[1])]
        else:
            p[1].append(VariableDeclarator(p[3]))
            p[0] = p[1]

    def p_normal_assignment(self, p):
        """
        assignment : postfix_expression '=' conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = Assignment(self._require_lvalue(p, 1), p[3])

    def p_addition_assignment(self, p):
        """
        assignment : postfix_expression ADDITION_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = AdditionAssignment(self._require_lvalue(p, 1), p[3])

    def p_subtraction_assignment(self, p):
        """
        assignment : postfix_expression SUBTRACTION_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = SubtractionAssignment(self._require_lvalue(p, 1), p[3])

    def p_multiplication_assignment(self, p):
        """
        assignment : postfix_expression MULTIPLICATION_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = MultiplicationAssignment(self._require_lvalue(p, 1), p[3])

    def p_division_assignment(self, p):
        """
        assignment : postfix_expression DIVISION_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = DivisionAssignment(self._require_lvalue(p, 1), p[3])

    def p_modulo_assignment(self, p):
        """
        assignment : postfix_expression MOD_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = ModuloAssignment(self._require_lvalue(p, 1), p[3])

    def p_bitwise_and_assignment(self, p):
        """
        assignment : postfix_expression BITWISE_AND_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseAndAssignment(self._require_lvalue(p, 1), p[3])

    def p_bitwise_or_assignment(self, p):
        """
        assignment : postfix_expression BITWISE_OR_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseOrAssignment(self._require_lvalue(p, 1), p[3])

    def p_bitwise_xor_assignment(self, p):
        """
        assignment : postfix_expression BITWISE_XOR_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseXorAssignment(self._require_lvalue(p, 1), p[3])

    def p_bitwise_left_assignment(self, p):
        """
        assignment : postfix_expression BITWISE_LEFT_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseLeftAssignment(self._require_lvalue(p, 1), p[3])

    def p_bitwise_right_assignment(self, p):
        """
        assignment : postfix_expression BITWISE_RIGHT_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseRightAssignment(self._require_lvalue(p, 1), p[3])

    def p_bitwise_unsigned_right_assignment(self, p):
        """
        assignment : postfix_expression BITWISE_UNSIGNED_RIGHT_ASSIGNMENT_OPERATOR conditional_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseUnsignedRightAssignment(self._require_lvalue(p, 1), p[3])

    def p_condition_statement(self, p):
        """
        condition_statement : if_statement else_statement
        """  # noqa: D200, D400, D212, D415
        p[0] = ConditionalBlock(p[1], [], p[2])

    def p_if_statement(self, p):
        """
        if_statement : IF_CONTROL '(' expression ')' non_null_statement
        """  # noqa: D200, D400, D212, D415
        p[0] = ConditionAndBlock(p[3], self._statement_block(p[5]))

    def p_else_statement(self, p):
        """
        else_statement : ELSE_CONTROL statement
                       |
        """  # noqa: D400, D212, D415, D205
        p[0] = None if len(p) == 1 else self._statement_block(p[2])


    def p_parenthesis_expression(self, p):
        """
        primary_expression : '(' expression ')'
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = ParenthesizedExpression(p[2])

    def p_expression(self, p):
        """
        expression : assignment
                   | conditional_expression
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = p[1]

    def p_conditional_expression(self, p):
        """
        conditional_expression : binary_expression
                               | binary_expression '?' conditional_expression ':' conditional_expression
        """  # noqa: D400, D212, D403, D415, D205
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = TernaryConditionalExpression(p[1], p[3], p[5])

    def p_binary_expression(self, p):
        """
        binary_expression : unary_expression
        """  # noqa: D200, D400, D212, D415
        p[0] = p[1]

    def p_binary_operator(self, p):
        """
        binary_expression : binary_expression GREATER_THAN binary_expression
                          | binary_expression GREATER_THAN_OR_EQUALS binary_expression
                          | binary_expression LESS_THAN binary_expression
                          | binary_expression LESS_THAN_OR_EQUALS binary_expression
                          | binary_expression AND binary_expression
                          | binary_expression NOT_EQUALS binary_expression
                          | binary_expression EQUALS binary_expression
                          | binary_expression OR binary_expression
                          | binary_expression ADD binary_expression
                          | binary_expression MINUS binary_expression
                          | binary_expression MULTIPLY binary_expression
                          | binary_expression DIVIDE binary_expression
                          | binary_expression BITWISE_OR binary_expression
                          | binary_expression BITWISE_XOR binary_expression
                          | binary_expression BITWISE_AND binary_expression
                          | binary_expression BITWISE_LEFT binary_expression
                          | binary_expression BITWISE_RIGHT binary_expression
                          | binary_expression BITWISE_UNSIGNED_RIGHT binary_expression
                          | binary_expression MOD binary_expression
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = BinaryOperatorExpression(p[1], p[3], p[2].binary)

    def p_unary_expression(self, p):
        """
        unary_expression : postfix_expression
                         | MINUS postfix_expression
                         | BITWISE_NOT postfix_expression
                         | NOT postfix_expression
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = p[1] if len(p) == 2 else UnaryOperatorExpression(p[2], p[1].unary)

    def p_return_statement(self, p: yacc.YaccProduction):
        """
        return_statement : RETURN ';'
                         | RETURN expression ';'
        """  # noqa: D400, D212, D415, D205
        if len(p) == 3:
            p[0] = ReturnStatement()
        elif len(p) == 4:
            expr = cast("Expression", p[2])
            p[0] = ReturnStatement(expr)

    def p_primary_expression(self, p):
        """
        primary_expression : function_call
                           | constant_expression
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = p[1]

    def p_constant_expression(self, p):
        """
        constant_expression : INT_VALUE
                            | FLOAT_VALUE
                            | STRING_VALUE
                            | OBJECTSELF_VALUE
                            | OBJECTINVALID_VALUE
                            | TRUE_VALUE
                            | FALSE_VALUE
                            | INT_HEX_VALUE
        """  # noqa: D400, D212, D415, D205
        if p.slice[1].type == "STRING_VALUE":
            try:
                encode_ncs_string(p[1].value)
            except UnicodeEncodeError as exc:
                raise CompileError(
                    "String literal contains a character not representable in NCS's one-byte encoding",
                    location=p.slice[1].location,
                ) from exc
        p[0] = p[1]

    def p_identifier_expression(self, p):
        """
        primary_expression : IDENTIFIER
        """
        p[0] = IdentifierExpression(p[1])

    def p_postfix_expression(self, p):
        """
        postfix_expression : primary_expression
                           | postfix_expression '.' IDENTIFIER
        """
        p[0] = p[1] if len(p) == 2 else MemberAccessExpression(p[1], p[3])

    def p_function_call(self, p):
        """
        function_call : IDENTIFIER '(' function_call_params ')'
        """  # noqa: D200, D400, D212, D415
        p[0] = FunctionCallExpression(p[1], p[3])

    def p_function_call_params(self, p):
        """
        function_call_params : argument_list
                             |
        """  # noqa: D400, D212, D415, D205
        p[0] = p[1] if len(p) == 2 else []

    def p_argument_list(self, p):
        """
        argument_list : expression
                      | argument_list ',' expression
        """  # noqa: D400, D212, D415, D205
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[1].append(p[3])
            p[0] = p[1]

    def p_any_data_type(self, p):
        """
        any_data_type : VOID_TYPE
                      | non_void_data_type
        """  # noqa: D400, D212, D415, D205
        p[0] = DynamicDataType(p[1]) if isinstance(p[1], DataType) else p[1]

    def p_non_void_data_type(self, p):
        """
        non_void_data_type : INT_TYPE
                           | FLOAT_TYPE
                           | OBJECT_TYPE
                           | EVENT_TYPE
                           | EFFECT_TYPE
                           | LOCATION_TYPE
                           | STRING_TYPE
                           | TALENT_TYPE
                           | VECTOR_TYPE
                           | STRUCT IDENTIFIER
        """  # noqa: D400, D212, D415, D205
        if len(p) == 3:
            p[0] = DynamicDataType(DataType.STRUCT, p[2].label)
        else:
            p[0] = DynamicDataType(p[1])

    def p_prefix_increment_expression(self, p):
        """
        unary_expression : INCREMENT postfix_expression
        """  # noqa: D200, D400, D212, D403, D415
        p[0] = PrefixIncrementExpression(self._require_lvalue(p, 2))

    def p_postfix_increment_expression(self, p):
        """
        postfix_expression : postfix_expression INCREMENT
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = PostfixIncrementExpression(self._require_lvalue(p, 1))

    def p_prefix_decrement_expression(self, p):
        """
        unary_expression : DECREMENT postfix_expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = PrefixDecrementExpression(self._require_lvalue(p, 2))

    def p_postfix_decrement_expression(self, p):
        """
        postfix_expression : postfix_expression DECREMENT
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = PostfixDecrementExpression(self._require_lvalue(p, 1))

    def p_vector_expression(self, p):
        """
        primary_expression : '[' ']'
                           | '[' vector_component ']'
                           | '[' vector_component ',' vector_component ']'
                           | '[' vector_component ',' vector_component ',' vector_component ']'
        """  # noqa: D400, D212, D403, D415, D205
        zero = FloatExpression(0.0)
        if len(p) == 3:
            p[0] = VectorExpression(zero, FloatExpression(0.0), FloatExpression(0.0))
        elif len(p) == 4:
            p[0] = VectorExpression(p[2], FloatExpression(0.0), FloatExpression(0.0))
        elif len(p) == 6:
            p[0] = VectorExpression(p[2], p[4], FloatExpression(0.0))
        else:
            p[0] = VectorExpression(p[2], p[4], p[6])

    def p_vector_component(self, p):
        """
        vector_component : FLOAT_VALUE
                         | IDENTIFIER
        """  # noqa: D400, D212, D415, D205
        p[0] = IdentifierExpression(p[1]) if isinstance(p[1], Identifier) else p[1]

    def p_switch_statement(self, p):
        """
        switch_statement : SWITCH_CONTROL '(' expression ')' statement
        """  # noqa: D200, D400, D212, D415
        p[0] = SwitchStatement(p[3], self._statement_block(p[5]))

    def p_expression_switch_label(self, p):
        """
        switch_label : CASE_CONTROL expression ':'
        """  # noqa: D200, D400, D212, D415
        p[0] = ExpressionSwitchLabel(p[2])

    def p_default_switch_label(self, p):
        """
        switch_label : DEFAULT_CONTROL ':'
        """  # noqa: D200, D400, D212, D415
        p[0] = DefaultSwitchLabel()

