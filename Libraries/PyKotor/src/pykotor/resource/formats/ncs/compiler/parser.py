"""NSS (NWScript) parser: PLY yacc grammar and AST construction."""

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
    FieldAccessExpression,
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
    ModuloAssignment,
    MultiplicationAssignment,
    NopStatement,
    PostfixDecrementExpression,
    PostfixIncrementExpression,
    PrefixDecrementExpression,
    PrefixIncrementExpression,
    ReturnStatement,
    StructDefinition,
    StructMember,
    SubtractionAssignment,
    SwitchBlock,
    SwitchStatement,
    TernaryConditionalExpression,
    UnaryOperatorExpression,
    VariableDeclarator,
    VariableInitializer,
    VectorExpression,
    WhileLoopBlock,
)
from pykotor.resource.formats.ncs.compiler.lexer import NssLexer

if TYPE_CHECKING:
    from pykotor.common.script import DataType, ScriptConstant, ScriptFunction
    from pykotor.resource.formats.ncs.compiler.classes import (
        Expression,
    )
else:
    from pykotor.common.script import DataType


class NssParser:
    """NSS (NWScript Source) parser.

    Parses tokenized NSS source code into an abstract syntax tree (AST) using
    recursive descent parsing. Handles includes, function definitions, statements,
    expressions, and control flow constructs.

    References:
    ----------
        Observed in retail KotOR I and TSL.
        PLY (Python Lex-Yacc) library for parser generation

    """

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
    ):
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
        (
            "right",
            "=",
            "ADDITION_ASSIGNMENT_OPERATOR",
            "SUBTRACTION_ASSIGNMENT_OPERATOR",
            "MULTIPLICATION_ASSIGNMENT_OPERATOR",
            "DIVISION_ASSIGNMENT_OPERATOR",
            "MOD_ASSIGNMENT_OPERATOR",
            "BITWISE_AND_ASSIGNMENT_OPERATOR",
            "BITWISE_OR_ASSIGNMENT_OPERATOR",
            "BITWISE_XOR_ASSIGNMENT_OPERATOR",
            "BITWISE_LEFT_ASSIGNMENT_OPERATOR",
            "BITWISE_RIGHT_ASSIGNMENT_OPERATOR",
            "BITWISE_UNSIGNED_RIGHT_ASSIGNMENT_OPERATOR",
        ),
        ("right", "?"),
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
        ("right", "UMINUS", "UPLUS", "BITWISE_NOT", "NOT"),
        ("left", "INCREMENT", "DECREMENT"),
    )

    def p_error(self, p) -> NoReturn:
        if p is None:
            raise CompileError("Syntax error: unexpected end of file")
        msg = f"Syntax error at line {p.lineno}, position {p.lexpos}, token='{p.value}'"
        raise CompileError(msg)

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
            p[0] = CodeRoot(
                constants=self.constants,
                functions=self.functions,
                library_lookup=self.library_lookup,
                library=self.library,
                max_include_depth=self.max_include_depth,
            )

    def p_code_root_object(self, p):
        """
        code_root_object : include_script
                         | typed_top_level_object
                         | const_global_variable_statement
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
        p[0] = IncludeScript(p[2], library=self.library)

    @staticmethod
    def _global_objects(
        data_type: DynamicDataType,
        identifier: Identifier,
        tail,
        *,
        is_const: bool,
    ):
        initializer, trailing_declarators = tail
        declarators = [
            VariableInitializer(identifier, initializer)
            if initializer is not None
            else VariableDeclarator(identifier),
            *trailing_declarators,
        ]
        objects = []
        for declarator in declarators:
            if isinstance(declarator, VariableInitializer):
                objects.append(
                    GlobalVariableInitialization(
                        declarator.identifier,
                        data_type,
                        declarator.expression,
                        is_const=is_const,
                    )
                )
            else:
                objects.append(
                    GlobalVariableDeclaration(
                        declarator.identifier,
                        data_type,
                        is_const=is_const,
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
                "  void can only be used as a function return type"
            )
        p[0] = self._global_objects(data_type, identifier, p[3], is_const=False)

    def p_const_global_variable_statement(self, p):
        """
        const_global_variable_statement : CONST const_data_type IDENTIFIER global_variable_tail ';'
        """  # noqa: D200, D400, D212, D415
        p[0] = self._global_objects(p[2], p[3], p[4], is_const=True)

    def p_global_variable_tail(self, p):
        """
        global_variable_tail : '=' expression global_variable_more
                             | global_variable_more
        """  # noqa: D400, D212, D415, D205
        if len(p) == 4:
            p[0] = (p[2], p[3])
        else:
            p[0] = (None, p[1])

    def p_global_variable_more(self, p):
        """
        global_variable_more : ',' variable_declarators
                             |
        """  # noqa: D400, D212, D415, D205
        p[0] = p[2] if len(p) == 3 else []

    def p_function_definition_params(self, p):
        """
        function_definition_params : function_definition_params ',' function_definition_param
                                   | function_definition_param
                                   |
        """  # noqa: D400, D212, D415, D205
        if len(p) == 4:
            cast("list", p[1]).append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 1:
            p[0] = []

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
        code_block : code_block statement
                   | statement
                   |
        """  # noqa: D400, D212, D415, D205
        if len(p) == 3:
            block: CodeBlock = p[1]
            block.add(p[2])
            p[0] = block
        elif len(p) == 2:
            block = CodeBlock()
            block.add(p[1])
            p[0] = block
        elif len(p) == 1:
            p[0] = CodeBlock()

    @staticmethod
    def _statement_block(statement) -> CodeBlock:
        if isinstance(statement, CodeBlock):
            return statement
        block = CodeBlock()
        block.add(statement)
        return block

    def p_while_loop(self, p):
        """
        while_loop : WHILE_CONTROL '(' expression ')' non_null_statement
        """  # noqa: D200, D400, D212, D415
        p[0] = WhileLoopBlock(p[3], self._statement_block(p[5]))

    def p_do_while_loop(self, p):
        """
        do_while_loop : DO_CONTROL non_null_statement WHILE_CONTROL '(' expression ')' ';'
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
        non_null_statement : declaration_statement
                           | condition_statement
                           | return_statement
                           | while_loop
                           | do_while_loop
                           | for_loop
                           | switch_statement
                           | break_statement
                           | continue_statement
                           | scoped_block
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = p[1]

    def p_nop_statement(self, p):
        """
        non_null_statement : NOP STRING_VALUE ';'
        """  # noqa: D200, D400, D403, D212, D415
        # p[2] is a StringExpression object from the lexer, access its .value attribute
        string_expr = p[2]
        p[0] = NopStatement(string_expr.value)

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

    def p_declaration_statement_const(self, p):
        """
        declaration_statement : CONST const_data_type variable_declarators ';'
        """  # noqa: D200, D400, D212, D415
        p[0] = DeclarationStatement(p[2], p[3], is_const=True)

    def p_variable_declarators(self, p):
        """
        variable_declarators : variable_declarators ',' variable_declarator
                             | variable_declarator
        """  # noqa: D400, D212, D415, D205
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_variable_declarator_no_initializer(self, p):
        """
        variable_declarator : IDENTIFIER
        """  # noqa: D200, D400, D212, D415
        p[0] = VariableDeclarator(p[1])

    def p_variable_declarator_initializer(self, p):
        """
        variable_declarator : IDENTIFIER '=' expression
        """  # noqa: D200, D400, D212, D415
        p[0] = VariableInitializer(p[1], p[3])

    def p_normal_assignment(self, p):
        """
        assignment : field_access '=' expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = Assignment(p[1], p[3])

    def p_addition_assignment(self, p):
        """
        assignment : field_access ADDITION_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = AdditionAssignment(p[1], p[3])

    def p_subtraction_assignment(self, p):
        """
        assignment : field_access SUBTRACTION_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = SubtractionAssignment(p[1], p[3])

    def p_multiplication_assignment(self, p):
        """
        assignment : field_access MULTIPLICATION_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = MultiplicationAssignment(p[1], p[3])

    def p_division_assignment(self, p):
        """
        assignment : field_access DIVISION_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = DivisionAssignment(p[1], p[3])

    def p_modulo_assignment(self, p):
        """
        assignment : field_access MOD_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = ModuloAssignment(p[1], p[3])

    def p_bitwise_and_assignment(self, p):
        """
        assignment : field_access BITWISE_AND_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseAndAssignment(p[1], p[3])

    def p_bitwise_or_assignment(self, p):
        """
        assignment : field_access BITWISE_OR_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseOrAssignment(p[1], p[3])

    def p_bitwise_xor_assignment(self, p):
        """
        assignment : field_access BITWISE_XOR_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseXorAssignment(p[1], p[3])

    def p_bitwise_left_assignment(self, p):
        """
        assignment : field_access BITWISE_LEFT_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseLeftAssignment(p[1], p[3])

    def p_bitwise_right_assignment(self, p):
        """
        assignment : field_access BITWISE_RIGHT_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseRightAssignment(p[1], p[3])

    def p_bitwise_unsigned_right_assignment(self, p):
        """
        assignment : field_access BITWISE_UNSIGNED_RIGHT_ASSIGNMENT_OPERATOR expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = BitwiseUnsignedRightAssignment(p[1], p[3])

    # region If Statement
    def p_condition_statement(self, p):
        """
        condition_statement : if_statement else_statement
        """  # noqa: D200, D400, D212, D415
        # ``else if`` is naturally represented as an ``else`` body containing
        # another conditional statement. This keeps the grammar small and gives
        # ``non_null_statement`` one clear meaning everywhere it is required.
        p[0] = ConditionalBlock(p[1], [], p[2])

    def p_if_statement(self, p):
        """
        if_statement : IF_CONTROL '(' expression ')' non_null_statement
        """  # noqa: D200, D400, D212, D415
        p[0] = ConditionAndBlock(p[3], self._statement_block(p[5]))

    def p_else_statement(self, p):
        """
        else_statement : ELSE_CONTROL non_null_statement
                       |
        """  # noqa: D400, D212, D415, D205
        p[0] = None if len(p) == 1 else self._statement_block(p[2])

    # endregion

    def p_parenthesis_expression(self, p):
        """
        expression : '(' expression ')'
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = p[2]

    def p_binary_operator(self, p):
        """
        expression : expression GREATER_THAN expression
                   | expression GREATER_THAN_OR_EQUALS expression
                   | expression LESS_THAN expression
                   | expression LESS_THAN_OR_EQUALS expression
                   | expression AND expression
                   | expression NOT_EQUALS expression
                   | expression EQUALS expression
                   | expression OR expression
                   | expression ADD expression
                   | expression MINUS expression
                   | expression MULTIPLY expression
                   | expression DIVIDE expression
                   | expression BITWISE_OR expression
                   | expression BITWISE_XOR expression
                   | expression BITWISE_AND expression
                   | expression BITWISE_LEFT expression
                   | expression BITWISE_RIGHT expression
                   | expression BITWISE_UNSIGNED_RIGHT expression
                   | expression MOD expression
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = BinaryOperatorExpression(p[1], p[3], p[2].binary)

    def p_ternary_expression(self, p):
        """
        expression : expression '?' expression ':' expression
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = TernaryConditionalExpression(p[1], p[3], p[5])

    def p_unary_expression(self, p):
        """
        expression : MINUS expression %prec UMINUS
                   | BITWISE_NOT expression
                   | NOT expression
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = UnaryOperatorExpression(p[2], p[1].unary)

    def p_unary_plus_expression(self, p):
        """
        expression : ADD expression %prec UPLUS
        """  # noqa: D200, D400, D212, D403, D415
        # Unary plus is a semantic no-op in NWScript. Returning the operand keeps
        # constant folding and runtime code generation identical to the operand.
        p[0] = p[2]

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

    def p_expression(self, p):
        """
        expression : function_call
                   | IDENTIFIER
                   | assignment
                   | constant_expression
        """  # noqa: D400, D212, D403, D415, D205
        p[0] = IdentifierExpression(p[1]) if isinstance(p[1], Identifier) else p[1]

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
        p[0] = p[1]

    def p_field_access_expression(self, p):
        """
        expression : field_access
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = FieldAccessExpression(p[1])

    def p_function_call(self, p):
        """
        function_call : IDENTIFIER '(' function_call_params ')'
        """  # noqa: D200, D400, D212, D415
        # Parsing records source syntax only. Whether this name denotes a user
        # function or a predefined engine routine is a semantic-resolution decision.
        p[0] = FunctionCallExpression(p[1], p[3])

    def p_function_call_params(self, p):
        """
        function_call_params : function_call_params ',' expression
                             | expression
                             |
        """  # noqa: D400, D212, D415, D205
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]
        elif len(p) == 1:
            p[0] = []

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

    def p_const_data_type(self, p):
        """
        const_data_type : INT_TYPE
                        | FLOAT_TYPE
                        | STRING_TYPE
        """  # noqa: D400, D212, D415, D205
        p[0] = DynamicDataType(p[1])

    def p_field_access(self, p):
        """
        field_access : IDENTIFIER
                     | IDENTIFIER '.' IDENTIFIER
                     | field_access '.' IDENTIFIER
        """  # noqa: D400, D212, D415, D205
        if len(p) == 2:
            p[0] = FieldAccess([p[1]])
        elif isinstance(p[1], Identifier):
            p[0] = FieldAccess([p[1], p[3]])
        else:
            p[0] = p[1]
            p[0].identifiers.append(p[3])

    def p_prefix_increment_expression(self, p):
        """
        expression : INCREMENT field_access
        """  # noqa: D200, D400, D212, D403, D415
        p[0] = PrefixIncrementExpression(p[2])

    def p_postfix_increment_expression(self, p):
        """
        expression : field_access INCREMENT
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = PostfixIncrementExpression(p[1])

    def p_prefix_decrement_expression(self, p):
        """
        expression : DECREMENT field_access
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = PrefixDecrementExpression(p[2])

    def p_postfix_decrement_expression(self, p):
        """
        expression : field_access DECREMENT
        """  # noqa: D200, D400, D403, D212, D415
        p[0] = PostfixDecrementExpression(p[1])

    def p_vector_expression(self, p):
        """
        expression : '[' ']'
                   | '[' FLOAT_VALUE ']'
                   | '[' FLOAT_VALUE ',' FLOAT_VALUE ']'
                   | '[' FLOAT_VALUE ',' FLOAT_VALUE ',' FLOAT_VALUE ']'
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

    # region Switch Statement
    def p_switch_statement(self, p):
        """
        switch_statement : SWITCH_CONTROL '(' expression ')' '{' switch_blocks '}'
        """  # noqa: D200, D400, D212, D415
        p[0] = SwitchStatement(p[3], p[6])

    def p_switch_blocks(self, p):
        """
        switch_blocks : switch_blocks switch_block
                      |
        """  # noqa: D400, D212, D415, D205
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        else:
            p[0] = []

    def p_switch_block(self, p):
        """
        switch_block : switch_labels block_statements
        """  # noqa: D200, D400, D212, D415
        p[0] = SwitchBlock(p[1], p[2])

    def p_switch_labels(self, p):
        """
        switch_labels : switch_labels switch_label
                      |
        """  # noqa: D400, D212, D415, D205
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        else:
            p[0] = []

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

    def p_block_statements(self, p):
        """
        block_statements : block_statements statement
                         |
        """  # noqa: D400, D212, D415, D205
        if len(p) == 3:
            p[1].append(p[2])
            p[0] = p[1]
        else:
            p[0] = []

    # endregion
