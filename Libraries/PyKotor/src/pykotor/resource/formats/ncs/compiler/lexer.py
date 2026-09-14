"""PLY lexer for NSS source."""

from __future__ import annotations

from typing import ClassVar

from ply import lex

from pykotor.common.script import DataType
from pykotor.resource.formats.ncs import NCSInstructionType
from pykotor.resource.formats.ncs.compiler.classes import (
    BinaryOperatorMapping,
    ControlKeyword,
    FloatExpression,
    Identifier,
    IntExpression,
    ObjectExpression,
    OperatorMapping,
    StringExpression,
    UnaryOperatorMapping,
)
from pykotor.resource.formats.ncs.compiler.numeric import parse_kotor_float_literal
from pykotor.resource.formats.ncs.compiler.source import (
    CompileError,
    SourceDocument,
    decode_nss_source,
    describe_token,
)


class NssLexer:
    """Tokenize NSS keywords, operators, identifiers, and literals."""

    def __init__(
        self,
        errorlog: lex.NullLogger = lex.NullLogger(),  # noqa: B008
        *,
        nowarn: bool = True,
        source_name: str = "<string>",
        source_encoding: str | None = None,
    ):
        self.source_name = source_name
        self.source_encoding = source_encoding
        self.source = SourceDocument(source_name, "")
        self.lexer: lex.Lexer = lex.lex(module=self, errorlog=errorlog, nowarn=nowarn)
        self._ply_input = self.lexer.input
        self._ply_token = self.lexer.token
        self.lexer.input = self.input
        self.lexer.token = self.token
        self.lexer.source = self.source
        self.lexer.begin("INITIAL")

    def input(self, source: str | bytes) -> None:
        """Start a fresh source file, even when the lexer instance is reused."""
        if isinstance(source, bytes):
            source = decode_nss_source(source, source_name=self.source_name, encoding=self.source_encoding)
        if not isinstance(source, str):
            raise TypeError("NSS source must be str or bytes")
        source = source.removeprefix("\ufeff")
        self.source = SourceDocument(self.source_name, source)
        self.lexer.source = self.source
        self.lexer.lineno = 1
        self.lexer.begin("INITIAL")
        self._ply_input(source)

    def token(self) -> lex.LexToken | None:
        """Keep the original lexeme separately from the parser's typed value."""
        token = self._ply_token()
        if token is not None:
            token.lexeme = self.source.text[token.lexpos:self.lexer.lexpos]
            token.location = self.source.location(token.lexpos)
        return token

    def t_error(self, token):
        """Turn lexical failures into the same located diagnostics as syntax errors."""
        message = (
            "Unterminated string literal"
            if token.value.startswith('"')
            else f"unexpected character {describe_token(token.value[0])}"
        )
        raise CompileError(message, location=self.source.location(token.lexpos))

    tokens: ClassVar[list[str]] = [
        "STRING_VALUE",
        "INT_VALUE",
        "FLOAT_VALUE",
        "IDENTIFIER",
        "INT_TYPE",
        "INT_HEX_VALUE",
        "FLOAT_TYPE",
        "OBJECT_TYPE",
        "VOID_TYPE",
        "EVENT_TYPE",
        "EFFECT_TYPE",
        "LOCATION_TYPE",
        "STRING_TYPE",
        "TALENT_TYPE",
        "VECTOR_TYPE",
        "ACTION_TYPE",
        "BREAK_CONTROL",
        "CASE_CONTROL",
        "DEFAULT_CONTROL",
        "DO_CONTROL",
        "ELSE_CONTROL",
        "SWITCH_CONTROL",
        "WHILE_CONTROL",
        "FOR_CONTROL",
        "IF_CONTROL",
        "TRUE_VALUE",
        "FALSE_VALUE",
        "OBJECTSELF_VALUE",
        "OBJECTINVALID_VALUE",
        "ADD",
        "MINUS",
        "MULTIPLY",
        "DIVIDE",
        "MOD",
        "EQUALS",
        "NOT_EQUALS",
        "GREATER_THAN",
        "LESS_THAN",
        "LESS_THAN_OR_EQUALS",
        "GREATER_THAN_OR_EQUALS",
        "AND",
        "OR",
        "NOT",
        "BITWISE_AND",
        "BITWISE_OR",
        "BITWISE_LEFT",
        "BITWISE_RIGHT",
        "BITWISE_UNSIGNED_RIGHT",
        "BITWISE_XOR",
        "BITWISE_NOT",
        "INCLUDE",
        "RETURN",
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
        "CONTINUE_CONTROL",
        "STRUCT",
        "INCREMENT",
        "DECREMENT",
    ]

    literals: ClassVar[list[str]] = [
        "{",
        "}",
        "(",
        ")",
        ";",
        "=",
        ",",
        ":",
        ".",
        "[",
        "]",
        "?",
    ]

    t_ignore: str = " \t"

    def t_NEWLINE(self, t):
        r"\r\n|\r|\n"  # noqa: D300, D400, D415
        t.lexer.lineno = self.source.location(t.lexer.lexpos).line

    def t_COMMENT(self, t):
        r"//[^\n]*"  # noqa: D300, D400, D415
        t.lexer.lineno = self.source.location(t.lexer.lexpos).line

    def t_MULTILINE_COMMENT(self, t):
        r"\/\*(\*(?!\/)|[^*])*\*\/"  # noqa: D300, D400, D415
        t.lexer.lineno = self.source.location(t.lexer.lexpos).line

    def t_UNTERMINATED_COMMENT(self, t):
        r"/\*(?:[^*]|\*(?!/))*\Z"
        raise CompileError("Unterminated block comment", location=self.source.location(t.lexpos))

    def t_INCLUDE(self, t):
        r"\#include"  # noqa: D300, D400, D415
        return t

    def t_OBJECTSELF_VALUE(self, t):
        r"OBJECT_SELF\b"  # noqa: D300, D400, D415
        t.value = ObjectExpression(0)
        return t

    def t_OBJECTINVALID_VALUE(self, t):
        r"OBJECT_INVALID\b"  # noqa: D300, D400, D415
        t.value = ObjectExpression(1)
        return t

    def t_TRUE_VALUE(self, t):
        r"TRUE\b"  # noqa: D300, D400, D415
        t.value = IntExpression(1)
        return t

    def t_FALSE_VALUE(self, t):
        r"FALSE\b"  # noqa: D300, D400, D415
        t.value = IntExpression(0)
        return t

    def t_BREAK_CONTROL(self, t):
        r"break\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.BREAK
        return t

    def t_CONTINUE_CONTROL(self, t):
        r"continue\b"  # noqa: D300, D400, D415
        return t

    def t_CASE_CONTROL(self, t):
        r"case\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.CASE
        return t

    def t_DEFAULT_CONTROL(self, t):
        r"default\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.DEFAULT
        return t

    def t_DO_CONTROL(self, t):
        r"do\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.DO
        return t

    def t_ELSE_CONTROL(self, t):
        r"else\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.ELSE
        return t

    def t_SWITCH_CONTROL(self, t):
        r"switch\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.SWITCH
        return t

    def t_WHILE_CONTROL(self, t):
        r"while\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.WHILE
        return t

    def t_FOR_CONTROL(self, t):
        r"for\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.FOR
        return t

    def t_IF_CONTROL(self, t):
        r"if\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.IF
        return t

    def t_RETURN(self, t):
        r"return\b"  # noqa: D300, D400, D415
        t.value = ControlKeyword.RETURN
        return t

    def t_STRUCT(self, t):
        r"struct\b"  # noqa: D300, D400, D415
        t.value = DataType.STRUCT
        return t

    def t_INT_TYPE(self, t):
        r"int\b"  # noqa: D300, D400, D415
        t.value = DataType.INT
        return t

    def t_FLOAT_TYPE(self, t):
        r"float\b"  # noqa: D300, D400, D415
        t.value = DataType.FLOAT
        return t

    def t_OBJECT_TYPE(self, t):
        r"object\b"  # noqa: D300, D400, D415
        t.value = DataType.OBJECT
        return t

    def t_VOID_TYPE(self, t):
        r"void\b"  # noqa: D300, D400, D415
        t.value = DataType.VOID
        return t

    def t_EVENT_TYPE(self, t):
        r"event\b"  # noqa: D300, D400, D415
        t.value = DataType.EVENT
        return t

    def t_EFFECT_TYPE(self, t):
        r"effect\b"  # noqa: D300, D400, D415
        t.value = DataType.EFFECT
        return t

    def t_LOCATION_TYPE(self, t):
        r"location\b"  # noqa: D300, D400, D415
        t.value = DataType.LOCATION
        return t

    def t_STRING_TYPE(self, t):
        r"string\b"  # noqa: D300, D400, D415
        t.value = DataType.STRING
        return t

    def t_TALENT_TYPE(self, t):
        r"talent\b"  # noqa: D300, D400, D415
        t.value = DataType.TALENT
        return t

    def t_ACTION_TYPE(self, t):
        r"action\b"  # noqa: D300, D400, D415
        t.value = DataType.ACTION
        return t

    def t_VECTOR_TYPE(self, t):
        r"vector\b"  # noqa: D300, D400, D415
        t.value = DataType.VECTOR
        return t


    def t_IDENTIFIER(self, t):
        "[a-zA-Z_]+[a-zA-Z0-9_]*"  # noqa: D300, D400, D415
        t.value = Identifier(t.value)
        return t

    @staticmethod
    def _decode_string_literal(value: str) -> str:
        """Decode newline escapes and discard other backslashes."""
        source = value[1:-1]
        result: list[str] = []
        index = 0
        while index < len(source):
            char = source[index]
            if char != "\\":
                result.append(char)
                index += 1
                continue

            if index + 1 < len(source) and source[index + 1] == "n":
                result.append("\n")
                index += 2
                continue

            index += 1

        return "".join(result)

    def t_STRING_VALUE(self, t):
        r'"[^"\n]*"'  # noqa: D300, D400, D415
        t.lexer.lineno = self.source.location(t.lexer.lexpos).line
        t.value = StringExpression(self._decode_string_literal(t.value))
        return t

    def t_FLOAT_VALUE(self, t):
        r"[0-9]+\.[0-9]*f?"  # noqa: D300, D400, D415
        literal = t.value[:-1] if t.value.endswith("f") else t.value
        t.value = FloatExpression(parse_kotor_float_literal(literal))
        return t

    def t_INT_HEX_VALUE(self, t):
        r"0[xX][0-9a-fA-F]+"  # noqa: D300, D400, D415
        t.value = IntExpression(int(t.value[2:], 16))
        return t

    def t_INT_VALUE(self, t):
        "[0-9]+"  # noqa: D300, D400, D415
        t.value = IntExpression(int(t.value))
        return t


    def t_INCREMENT(self, t):
        r"\+\+"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[
                UnaryOperatorMapping(NCSInstructionType.INCISP, DataType.INT),
            ],
            binary=[],
        )
        return t

    def t_DECREMENT(self, t):
        r"\-\-"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[
                UnaryOperatorMapping(NCSInstructionType.DECISP, DataType.INT),
            ],
            binary=[],
        )
        return t

    def t_ADDITION_ASSIGNMENT_OPERATOR(self, t):
        r"\+\="  # noqa: D300, D400, D415
        return t

    def t_SUBTRACTION_ASSIGNMENT_OPERATOR(self, t):
        r"\-\="  # noqa: D300, D400, D415
        return t

    def t_MULTIPLICATION_ASSIGNMENT_OPERATOR(self, t):
        r"\*\="  # noqa: D300, D400, D415
        return t

    def t_DIVISION_ASSIGNMENT_OPERATOR(self, t):
        r"/\="  # noqa: D300, D400, D415
        return t

    def t_MOD_ASSIGNMENT_OPERATOR(self, t):
        r"%\="  # noqa: D300, D400, D415
        return t

    def t_BITWISE_AND_ASSIGNMENT_OPERATOR(self, t):
        r"&\="  # noqa: D300, D400, D415
        return t

    def t_BITWISE_OR_ASSIGNMENT_OPERATOR(self, t):
        r"\|\="  # noqa: D300, D400, D415
        return t

    def t_BITWISE_XOR_ASSIGNMENT_OPERATOR(self, t):
        r"\^\="  # noqa: D300, D400, D415
        return t

    def t_BITWISE_LEFT_ASSIGNMENT_OPERATOR(self, t):
        r"<<\="  # noqa: D300, D400, D415
        return t

    def t_BITWISE_RIGHT_ASSIGNMENT_OPERATOR(self, t):
        r">>\="  # noqa: D300, D400, D415
        return t

    def t_BITWISE_UNSIGNED_RIGHT_ASSIGNMENT_OPERATOR(self, t):
        r">>>\="  # noqa: D300, D400, D415
        return t

    def t_BITWISE_LEFT(self, t):
        "<<"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.SHLEFTII, DataType.INT, DataType.INT, DataType.INT
                ),
            ],
        )
        return t

    def t_BITWISE_UNSIGNED_RIGHT(self, t):
        ">>>"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.USHRIGHTII, DataType.INT, DataType.INT, DataType.INT
                ),
            ],
        )
        return t

    def t_BITWISE_RIGHT(self, t):
        ">>"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.SHRIGHTII, DataType.INT, DataType.INT, DataType.INT
                ),
            ],
        )
        return t

    def t_ADD(self, t):
        r"\+"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.ADDII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.ADDIF, DataType.FLOAT, DataType.INT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.ADDFI, DataType.FLOAT, DataType.FLOAT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.ADDFF, DataType.FLOAT, DataType.FLOAT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.ADDVV, DataType.VECTOR, DataType.VECTOR, DataType.VECTOR
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.ADDSS, DataType.STRING, DataType.STRING, DataType.STRING
                ),
            ],
        )
        return t

    def t_MINUS(self, t):
        "-"  # noqa: D300, D415, D400
        t.value = OperatorMapping(
            unary=[
                UnaryOperatorMapping(NCSInstructionType.NEGI, DataType.INT),
                UnaryOperatorMapping(NCSInstructionType.NEGF, DataType.FLOAT),
            ],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.SUBII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.SUBIF, DataType.FLOAT, DataType.INT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.SUBFI, DataType.FLOAT, DataType.FLOAT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.SUBFF, DataType.FLOAT, DataType.FLOAT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.SUBVV, DataType.VECTOR, DataType.VECTOR, DataType.VECTOR
                ),
            ],
        )
        return t

    def t_MULTIPLY(self, t):
        r"\*"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.MULII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.MULIF, DataType.FLOAT, DataType.INT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.MULFI, DataType.FLOAT, DataType.FLOAT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.MULFF, DataType.FLOAT, DataType.FLOAT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.MULVF, DataType.VECTOR, DataType.VECTOR, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.MULFV, DataType.VECTOR, DataType.FLOAT, DataType.VECTOR
                ),
            ],
        )
        return t

    def t_DIVIDE(self, t):
        "/"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.DIVII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.DIVIF, DataType.FLOAT, DataType.INT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.DIVFI, DataType.FLOAT, DataType.FLOAT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.DIVFF, DataType.FLOAT, DataType.FLOAT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.DIVVF, DataType.VECTOR, DataType.VECTOR, DataType.FLOAT
                ),
            ],
        )
        return t

    def t_MOD(self, t):
        r"\%"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.MODII, DataType.INT, DataType.INT, DataType.INT
                ),
            ],
        )
        return t

    def t_EQUALS(self, t):
        r"\=\="  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.EQUALII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.EQUALFF, DataType.INT, DataType.FLOAT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.EQUALOO, DataType.INT, DataType.OBJECT, DataType.OBJECT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.EQUALSS, DataType.INT, DataType.STRING, DataType.STRING
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.EQUALEFFEFF, DataType.INT, DataType.EFFECT, DataType.EFFECT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.EQUALEVTEVT, DataType.INT, DataType.EVENT, DataType.EVENT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.EQUALLOCLOC, DataType.INT, DataType.LOCATION, DataType.LOCATION
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.EQUALTALTAL, DataType.INT, DataType.TALENT, DataType.TALENT
                ),
            ],
        )
        return t

    def t_NOT_EQUALS(self, t):
        r"\!="  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.NEQUALII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.NEQUALFF, DataType.INT, DataType.FLOAT, DataType.FLOAT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.NEQUALOO, DataType.INT, DataType.OBJECT, DataType.OBJECT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.NEQUALSS, DataType.INT, DataType.STRING, DataType.STRING
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.NEQUALEFFEFF, DataType.INT, DataType.EFFECT, DataType.EFFECT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.NEQUALEVTEVT, DataType.INT, DataType.EVENT, DataType.EVENT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.NEQUALLOCLOC, DataType.INT, DataType.LOCATION, DataType.LOCATION
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.NEQUALTALTAL, DataType.INT, DataType.TALENT, DataType.TALENT
                ),
            ],
        )
        return t

    def t_GREATER_THAN_OR_EQUALS(self, t):
        r">\="  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.GEQII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.GEQFF, DataType.INT, DataType.FLOAT, DataType.FLOAT
                ),
            ],
        )
        return t

    def t_GREATER_THAN(self, t):
        ">"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.GTII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.GTFF, DataType.INT, DataType.FLOAT, DataType.FLOAT
                ),
            ],
        )
        return t

    def t_LESS_THAN_OR_EQUALS(self, t):
        r"\<="  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.LEQII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.LEQFF, DataType.INT, DataType.FLOAT, DataType.FLOAT
                ),
            ],
        )
        return t

    def t_LESS_THAN(self, t):
        r"\<"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.LTII, DataType.INT, DataType.INT, DataType.INT
                ),
                BinaryOperatorMapping(
                    NCSInstructionType.LTFF, DataType.INT, DataType.FLOAT, DataType.FLOAT
                ),
            ],
        )
        return t

    def t_AND(self, t):
        "&&"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.LOGANDII, DataType.INT, DataType.INT, DataType.INT
                ),
            ],
        )
        return t

    def t_OR(self, t):
        r"\|\|"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.LOGORII, DataType.INT, DataType.INT, DataType.INT
                ),
            ],
        )
        return t

    def t_NOT(self, t):
        r"\!"  # noqa: D300, D400
        t.value = OperatorMapping(
            unary=[
                UnaryOperatorMapping(NCSInstructionType.NOTI, DataType.INT),
            ],
            binary=[],
        )
        return t

    def t_BITWISE_AND(self, t):
        "&"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.BOOLANDII, DataType.INT, DataType.INT, DataType.INT
                ),
            ],
        )
        return t

    def t_BITWISE_OR(self, t):
        r"\|"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.INCORII, DataType.INT, DataType.INT, DataType.INT
                ),
            ],
        )
        return t

    def t_BITWISE_XOR(self, t):
        r"\^"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[],
            binary=[
                BinaryOperatorMapping(
                    NCSInstructionType.EXCORII, DataType.INT, DataType.INT, DataType.INT
                ),
            ],
        )
        return t

    def t_BITWISE_NOT(self, t):
        r"\~"  # noqa: D300, D400, D415
        t.value = OperatorMapping(
            unary=[
                UnaryOperatorMapping(NCSInstructionType.COMPI, DataType.INT),
            ],
            binary=[],
        )
        return t

