from __future__ import annotations

import os
import pathlib
import sys
import unittest

THIS_SCRIPT_PATH = pathlib.Path(__file__).resolve()
PYKOTOR_PATH = THIS_SCRIPT_PATH.parents[3].resolve()
UTILITY_PATH = THIS_SCRIPT_PATH.parents[5].joinpath("Utility", "src").resolve()


def add_sys_path(p: pathlib.Path):
    working_dir = str(p)
    if working_dir not in sys.path:
        sys.path.append(working_dir)


if PYKOTOR_PATH.joinpath("pykotor").exists():
    add_sys_path(PYKOTOR_PATH)
if UTILITY_PATH.joinpath("utility").exists():
    add_sys_path(UTILITY_PATH)


from pykotor.common.geometry import Vector3
from pykotor.common.scriptdefs import KOTOR_CONSTANTS, KOTOR_FUNCTIONS
from pykotor.resource.formats.ncs import NCS, NCSInstructionType
from pykotor.resource.formats.ncs.compiler.classes import (
    DEFAULT_MAX_INCLUDE_DEPTH,
    MAX_FUNCTION_PARAMETERS,
    CompileError,
    ExpressionStatement,
    FunctionCallExpression,
    FunctionDefinition,
)
from pykotor.resource.formats.ncs.compiler.interpreter import Interpreter
from pykotor.resource.formats.ncs.compiler.lexer import NssLexer
from pykotor.resource.formats.ncs.compiler.parser import NssParser
from utility.system.path import Path

K1_PATH: str | None = os.environ.get("K1_PATH")
K2_PATH: str | None = os.environ.get("K2_PATH")


class TestNSSCompiler(unittest.TestCase):
    def compile(
        self,
        script: str,
        library: dict[str, bytes] | None = None,
        library_lookup: list[str | Path] | list[str] | list[Path] | str | Path | None = None,
        *,
        max_include_depth: int = DEFAULT_MAX_INCLUDE_DEPTH,
    ) -> NCS:
        if library is None:
            library = {}
        nssLexer = NssLexer()
        nssParser = NssParser(
            library=library,
            constants=KOTOR_CONSTANTS,
            functions=KOTOR_FUNCTIONS,
            library_lookup=library_lookup,
            max_include_depth=max_include_depth,
        )

        parser = nssParser.parser
        t = parser.parse(script, tracking=True)

        ncs = NCS()
        t.compile(ncs)
        return ncs

    # region Engine Call
    def test_enginecall(self):
        ncs = self.compile(
            """
            void main()
            {
                object oExisting = GetExitingObject();
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))

        self.assertEqual("GetExitingObject", interpreter.action_snapshots[0].function_name)
        self.assertEqual([], interpreter.action_snapshots[0].arg_values)

    def test_enginecall_return_value(self):
        ncs = self.compile(
            """
            void main()
            {
                int inescapable = GetAreaUnescapable();
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.set_mock("GetAreaUnescapable", lambda: 10)
        interpreter.run()

        self.assertEqual(10, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_enginecall_with_params(self):
        ncs = self.compile(
            """
            void main()
            {
                string tag = "something";
                int n = 15;
                object oSomething = GetObjectByTag(tag, n);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))

        self.assertEqual("GetObjectByTag", interpreter.action_snapshots[0].function_name)
        self.assertEqual(["something", 15], interpreter.action_snapshots[0].arg_values)

    def test_enginecall_with_default_params(self):
        ncs = self.compile(
            """
            void main()
            {
                string tag = "something";
                object oSomething = GetObjectByTag(tag);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

    def test_enginecall_default_expansion_does_not_mutate_ast(self):
        parser = NssParser(
            library={},
            constants=KOTOR_CONSTANTS,
            functions=KOTOR_FUNCTIONS,
        )
        root = parser.parser.parse(
            'void main() { GetObjectByTag("tag"); }',
            lexer=NssLexer().lexer,
            tracking=True,
        )

        function = next(
            obj for obj in root.objects if isinstance(obj, FunctionDefinition)
        )
        statement = function.block.statements[0]
        self.assertIsInstance(statement, ExpressionStatement)
        call = statement.expression
        self.assertIsInstance(call, FunctionCallExpression)
        self.assertEqual(1, len(call.arguments))

        first_ncs = NCS()
        root.compile(first_ncs)
        self.assertEqual(1, len(call.arguments))

        second_ncs = NCS()
        root.compile(second_ncs)
        self.assertEqual(1, len(call.arguments))
        self.assertEqual(
            [instruction.ins_type for instruction in first_ncs.instructions],
            [instruction.ins_type for instruction in second_ncs.instructions],
        )

    def test_enginecall_with_missing_params(self):
        script = """
            void main()
            {
                string tag = "something";
                object oSomething = GetObjectByTag();
            }
        """

        self.assertRaises(CompileError, self.compile, script)

    def test_enginecall_with_too_many_params(self):
        script = """
            void main()
            {
                string tag = "something";
                object oSomething = GetObjectByTag("", 0, "shouldnotbehere");
            }
        """

        self.assertRaises(CompileError, self.compile, script)

    def test_enginecall_delay_command_1(self):
        ncs = self.compile(
            """
            void main()
            {
                object oFirstPlayer = GetFirstPC();
                DelayCommand(1.0, GiveXPToCreature(oFirstPlayer, 9001));
            }
        """
        )

    def test_action_parameter_accepts_void_engine_call(self):
        self.compile(
            """
            void main()
            {
                AssignCommand(OBJECT_SELF, ActionWait(1.0));
            }
            """
        )

    def test_action_parameter_accepts_void_user_call(self):
        self.compile(
            """
            void DoSomething()
            {
                PrintString("called");
            }

            void main()
            {
                AssignCommand(OBJECT_SELF, DoSomething());
            }
            """
        )

    def test_action_parameter_rejects_nonvoid_engine_call(self):
        script = """
            void main()
            {
                AssignCommand(OBJECT_SELF, Random(10));
            }
        """

        with self.assertRaisesRegex(CompileError, "ACTION parameter.*void expression"):
            self.compile(script)

    def test_action_parameter_rejects_nonvoid_user_call(self):
        script = """
            int GetValue()
            {
                return 7;
            }

            void main()
            {
                AssignCommand(OBJECT_SELF, GetValue());
            }
        """

        with self.assertRaisesRegex(CompileError, "ACTION parameter.*void expression"):
            self.compile(script)

    def test_user_function_cannot_collide_with_engine_function(self):
        script = """
            int Random(int nMaxInteger)
            {
                return 7;
            }

            void main()
            {
                int value = Random(1);
            }
        """

        with self.assertRaisesRegex(
            CompileError,
            "conflicts with a predefined engine function",
        ):
            self.compile(script)

    def test_engine_call_uses_semantic_resolution(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = Random(5);
            }
            """
        )

        action = next(
            instruction
            for instruction in ncs.instructions
            if instruction.ins_type == NCSInstructionType.ACTION
        )
        self.assertEqual(
            next(i for i, function in enumerate(KOTOR_FUNCTIONS) if function.name == "Random"),
            action.args[0],
        )

    def test_enginecall_GetFirstObjectInShape_defaults(self):
        # Tests defaults for (int, int, vector)
        ncs = self.compile(
            """
            void main()
            {
                int nShape = SHAPE_CUBE;
                float fSize = 0.0;
                location lTarget;
                GetFirstObjectInShape(nShape, fSize, lTarget);
            }
        """
        )

    def test_enginecall_GetFactionEqual(self):
        # Tests defaults for (object)
        ncs = self.compile(
            """
            void main()
            {
                object oFirst;
                GetFactionEqual(oFirst);
            }
        """
        )

    # endregion

    # region Operators
    def test_addop_int_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 10 + 5;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(15, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_addop_float_float(self):
        ncs = self.compile(
            """
            void main()
            {
                float value = 10.0 + 5.0;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(15.0, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_addop_string_string(self):
        ncs = self.compile(
            """
            void main()
            {
                string value = "abc" + "def";
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual("abcdef", interpreter.stack_snapshots[-4].stack[-1].value)

    def test_subop_int_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 10 - 5;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(5, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_subop_float_float(self):
        ncs = self.compile(
            """
            void main()
            {
                float value = 10.0 - 5.0;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(5.0, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_mulop_int_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 10 * 5;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(50, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_mulop_float_float(self):
        ncs = self.compile(
            """
            void main()
            {
                float a = 10.0 * 5.0;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(50.0, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_divop_int_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 10 / 5;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_divop_float_float(self):
        ncs = self.compile(
            """
            void main()
            {
                float a = 10.0 / 5.0;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2.0, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_modop_int_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 10 % 3;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_negop_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = -10;
                PrintInteger(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(-10, interpreter.action_snapshots[0].arg_values[0])

    def test_negop_float(self):
        ncs = self.compile(
            """
            void main()
            {
                float a = -10.0;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(-10.0, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_bidmas(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 2 + (5 * ((0)) + 5) * 3 + 2 - (2 + (2 * 4 - 12 / 2)) / 2;
                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(17, interpreter.action_snapshots[0].arg_values[0])

    def test_op_with_variables(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 10;
                int b = 5;
                int c = a * b * a;
                int d = 10 * 5 * 10;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(500, interpreter.stack_snapshots[-4].stack[-1].value)
        self.assertEqual(500, interpreter.stack_snapshots[-4].stack[-2].value)

    def test_mixed_int_float_arithmetic_promotes_to_float(self):
        ncs = self.compile(
            """
            void main()
            {
                int i = 8;
                float f = 2.0;
                float add = i + f;
                float sub = i - f;
                float mul = i * f;
                float div = i / f;
            }
            """
        )

        emitted = [instruction.ins_type for instruction in ncs.instructions]
        self.assertIn(NCSInstructionType.ADDIF, emitted)
        self.assertIn(NCSInstructionType.SUBIF, emitted)
        self.assertIn(NCSInstructionType.MULIF, emitted)
        self.assertIn(NCSInstructionType.DIVIF, emitted)

    def test_promoted_int_float_result_cannot_initialize_int(self):
        source = """
            void main()
            {
                int value = 1 + 2.0;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_float_vector_division_is_rejected(self):
        source = """
            void main()
            {
                vector v = [1.0, 2.0, 3.0];
                vector result = 2.0 / v;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    # test_addop_vector_vector
    # test_addop_int_float
    # test_addop_float_int
    # test_subop_int_float
    # test_subop_float_int
    # test_subop_vector_vector
    # test_mulop_int_float
    # test_mulop_float_int
    # test_mulop_float_vector
    # test_mulop_vector_float
    # test_divop_int_float
    # test_divop_float_int
    # test_divop_vector_float

    # endregion

    # region Logical Operator
    def test_unary_plus_int_and_float(self):
        ncs = self.compile(
            """
            void main()
            {
                int integerValue = +7;
                float floatValue = +2.5;
                int precedence = +1 * 3 + 4;
                PrintInteger(integerValue);
                PrintFloat(floatValue);
                PrintInteger(precedence);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(7, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(2.5, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(7, interpreter.action_snapshots[-1].arg_values[0])

    def test_not_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = !1;
                PrintInteger(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-1].arg_values[0])

    def test_logical_and_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 0 && 0;
                int b = 1 && 0;
                int c = 1 && 1;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.stack_snapshots[-4].stack[-3].value)
        self.assertEqual(0, interpreter.stack_snapshots[-4].stack[-2].value)
        self.assertEqual(1, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_logical_or_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 0 || 0;
                int b = 1 || 0;
                int c = 1 || 1;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.stack_snapshots[-4].stack[-3].value)
        self.assertEqual(1, interpreter.stack_snapshots[-4].stack[-2].value)
        self.assertEqual(1, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_logical_and_short_circuits_runtime_rhs(self):
        ncs = self.compile(
            """
            int touched = 0;

            int touch()
            {
                touched = 1;
                return 1;
            }

            void main()
            {
                int lhs = 0;
                int value = lhs && touch();
                PrintInteger(touched);
                PrintInteger(value);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(0, interpreter.action_snapshots[-1].arg_values[0])

    def test_logical_or_short_circuits_runtime_rhs(self):
        ncs = self.compile(
            """
            int touched = 0;

            int touch()
            {
                touched = 1;
                return 0;
            }

            void main()
            {
                int lhs = 7;
                int value = lhs || touch();
                PrintInteger(touched);
                PrintInteger(value);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-2].arg_values[0])
        # BioWare leaves the original non-zero lhs as the short-circuit result.
        self.assertEqual(7, interpreter.action_snapshots[-1].arg_values[0])

    def test_logical_short_circuit_evaluates_rhs_when_required(self):
        ncs = self.compile(
            """
            int touched = 0;

            int touch()
            {
                touched += 1;
                return 1;
            }

            void main()
            {
                int true_lhs = 1;
                int false_lhs = 0;
                int and_value = true_lhs && touch();
                int or_value = false_lhs || touch();
                PrintInteger(touched);
                PrintInteger(and_value);
                PrintInteger(or_value);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-1].arg_values[0])

    def test_logical_equals(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1 == 1;
                int b = "a" == "b";
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.stack_snapshots[-4].stack[-2].value)
        self.assertEqual(0, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_logical_notequals_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1 != 1;
                int b = 1 != 2;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.stack_snapshots[-4].stack[-2].value)
        self.assertEqual(1, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_struct_equality_uses_sized_struct_opcode(self):
        ncs = self.compile(
            """
            struct Pair
            {
                int first;
                int second;
            };

            void main()
            {
                struct Pair left;
                struct Pair right;
                int same = left == right;
                int different = left != right;
            }
            """
        )

        equal = [i for i in ncs.instructions if i.ins_type == NCSInstructionType.EQUALTT]
        not_equal = [i for i in ncs.instructions if i.ins_type == NCSInstructionType.NEQUALTT]
        self.assertEqual([[8]], [instruction.args for instruction in equal])
        self.assertEqual([[8]], [instruction.args for instruction in not_equal])

    def test_vector_equality_uses_sized_struct_opcode(self):
        ncs = self.compile(
            """
            void main()
            {
                vector left;
                vector right;
                int same = left == right;
                int different = left != right;
            }
            """
        )

        equal = [i for i in ncs.instructions if i.ins_type == NCSInstructionType.EQUALTT]
        not_equal = [i for i in ncs.instructions if i.ins_type == NCSInstructionType.NEQUALTT]
        self.assertEqual([[12]], [instruction.args for instruction in equal])
        self.assertEqual([[12]], [instruction.args for instruction in not_equal])

    def test_engine_structure_equality_uses_target_opcodes(self):
        ncs = self.compile(
            """
            void main()
            {
                effect effectA;
                effect effectB;
                event eventA;
                event eventB;
                location locationA;
                location locationB;
                talent talentA;
                talent talentB;

                int effectSame = effectA == effectB;
                int eventDifferent = eventA != eventB;
                int locationSame = locationA == locationB;
                int talentDifferent = talentA != talentB;
            }
            """
        )

        emitted = [instruction.ins_type for instruction in ncs.instructions]
        self.assertIn(NCSInstructionType.EQUALEFFEFF, emitted)
        self.assertIn(NCSInstructionType.NEQUALEVTEVT, emitted)
        self.assertIn(NCSInstructionType.EQUALLOCLOC, emitted)
        self.assertIn(NCSInstructionType.NEQUALTALTAL, emitted)

    def test_different_named_structs_cannot_be_compared(self):
        source = """
            struct LeftType
            {
                int value;
            };

            struct RightType
            {
                int value;
            };

            void main()
            {
                struct LeftType left;
                struct RightType right;
                int same = left == right;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    # endregion

    # region Relational Operator
    def test_compare_greaterthan_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 10 > 1;
                int b = 10 > 10;
                int c = 10 > 20;

                PrintInteger(a);
                PrintInteger(b);
                PrintInteger(c);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(0, interpreter.action_snapshots[-1].arg_values[0])

    def test_compare_greaterthanorequal_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 10 >= 1;
                int b = 10 >= 10;
                int c = 10 >= 20;

                PrintInteger(a);
                PrintInteger(b);
                PrintInteger(c);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(0, interpreter.action_snapshots[-1].arg_values[0])

    def test_compare_lessthan_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 10 < 1;
                int b = 10 < 10;
                int c = 10 < 20;

                PrintInteger(a);
                PrintInteger(b);
                PrintInteger(c);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-1].arg_values[0])

    def test_compare_lessthanorequal_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 10 <= 1;
                int b = 10 <= 10;
                int c = 10 <= 20;

                PrintInteger(a);
                PrintInteger(b);
                PrintInteger(c);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-1].arg_values[0])

    # endregion

    # region Bitwise Operator
    def test_bitwise_or_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 5 | 2;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(7, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_bitwise_xor_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 7 ^ 2;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(5, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_bitwise_not_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = ~1;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(-2, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_bitwise_and_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 7 & 2;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_bitwise_shiftleft_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 7 << 2;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(28, interpreter.stack_snapshots[-4].stack[-1].value)

    def test_bitwise_shiftright_op(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 7 >> 2;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.stack_snapshots[-4].stack[-1].value)

    # endregion

    # region Assignment
    def test_assignment(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1;
                a = 4;

                PrintInteger(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(4, interpreter.action_snapshots[0].arg_values[0])

    def test_assignment_complex(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1;
                a = a * 2 + 8;

                PrintInteger(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(10, interpreter.action_snapshots[0].arg_values[0])

    def test_assignment_string_constant(self):
        ncs = self.compile(
            """
            void main()
            {
                string a = "A";

                PrintString(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual("A", interpreter.action_snapshots[0].arg_values[0])

    def test_assignment_string_enginecall(self):
        ncs = self.compile(
            """
            void main()
            {
                string a = GetGlobalString("A");

                PrintString(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.set_mock("GetGlobalString", lambda identifier: identifier)
        interpreter.run()

        self.assertEqual("A", interpreter.action_snapshots[-1].arg_values[0])

    def test_addition_assignment_int_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 1;
                value += 2;

                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        snap = interpreter.action_snapshots[-1]
        self.assertEqual("PrintInteger", snap.function_name)
        self.assertEqual(3, snap.arg_values[0])

    def test_addition_assignment_int_float(self):
        source = """
            void main()
            {
                int value = 1;
                value += 2.0;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_addition_assignment_float_float(self):
        ncs = self.compile(
            """
            void main()
            {
                float value = 1.0;
                value += 2.0;

                PrintFloat(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(3.0, interpreter.action_snapshots[-1].arg_values[0])

    def test_addition_assignment_float_int(self):
        ncs = self.compile(
            """
            void main()
            {
                float value = 1.0;
                value += 2;

                PrintFloat(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        snap = interpreter.action_snapshots[-1]
        self.assertEqual("PrintFloat", snap.function_name)
        self.assertEqual(3.0, snap.arg_values[0])

    def test_addition_assignment_string_string(self):
        ncs = self.compile(
            """
            void main()
            {
                string value = "a";
                value += "b";

                PrintString(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        snap = interpreter.action_snapshots[-1]
        self.assertEqual("PrintString", snap.function_name)
        self.assertEqual("ab", snap.arg_values[0])

    def test_subtraction_assignment_int_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 10;
                value -= 2 * 2;

                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        snap = interpreter.action_snapshots[-1]
        self.assertEqual("PrintInteger", snap.function_name)
        self.assertEqual([6], snap.arg_values)

    def test_subtraction_assignment_int_float(self):
        source = """
            void main()
            {
                int value = 10;
                value -= 2.0;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_subtraction_assignment_float_float(self):
        ncs = self.compile(
            """
            void main()
            {
                float value = 10.0;
                value -= 2.0;

                PrintFloat(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        snap = interpreter.action_snapshots[-1]
        self.assertEqual("PrintFloat", snap.function_name)
        self.assertEqual(8.0, snap.arg_values[0])

    def test_subtraction_assignment_float_int(self):
        ncs = self.compile(
            """
            void main()
            {
                float value = 10.0;
                value -= 2;

                PrintFloat(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(8.0, interpreter.action_snapshots[-1].arg_values[0])

    def test_multiplication_assignment(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 10;
                value *= 2 * 2;

                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        snap = interpreter.action_snapshots[-1]
        self.assertEqual("PrintInteger", snap.function_name)
        self.assertEqual([40], snap.arg_values)

    def test_division_assignment(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 12;
                value /= 2 * 2;

                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        snap = interpreter.action_snapshots[-1]
        self.assertEqual("PrintInteger", snap.function_name)
        self.assertEqual([3], snap.arg_values)

    def test_multiplication_assignment_int_float_rejected(self):
        source = """
            void main()
            {
                int value = 3;
                value *= 2.0;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_division_assignment_int_float_rejected(self):
        source = """
            void main()
            {
                int value = 12;
                value /= 2.0;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    # endregion

    # region Switch Statements
    def test_switch_no_breaks(self):
        ncs = self.compile(
            """
            void main()
            {
                switch (2)
                {
                    case 1:
                        PrintInteger(1);
                    case 2:
                        PrintInteger(2);
                    case 3:
                        PrintInteger(3);
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(2, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(3, interpreter.action_snapshots[1].arg_values[0])

    def test_switch_jump_over(self):
        ncs = self.compile(
            """
            void main()
            {
                switch (4)
                {
                    case 1:
                        PrintInteger(1);
                    case 2:
                        PrintInteger(2);
                    case 3:
                        PrintInteger(3);
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, len(interpreter.action_snapshots))

    def test_switch_with_breaks(self):
        ncs = self.compile(
            """
            void main()
            {
                switch (3)
                {
                    case 1:
                        PrintInteger(1);
                        break;
                    case 2:
                        PrintInteger(2);
                        break;
                    case 3:
                        PrintInteger(3);
                        break;
                    case 4:
                        PrintInteger(4);
                        break;
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(3, interpreter.action_snapshots[0].arg_values[0])

    def test_switch_return_uses_normal_function_return_lowering(self):
        ncs = self.compile(
            """
            int choose(int value)
            {
                switch (value)
                {
                    case 1:
                        return 10;
                    default:
                        return 20;
                }

                // BioWare's return-path analysis does not treat switch as proof
                // that every control path returns.
                return 0;
            }

            void main()
            {
                PrintInteger(choose(1));
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(10, interpreter.action_snapshots[-1].arg_values[0])

    def test_switch_return_unwinds_discriminant_and_local(self):
        ncs = self.compile(
            """
            int choose(int value)
            {
                switch (value)
                {
                    default:
                        break;
                    case 1:
                        int local = 41;
                        return local + 1;
                }

                return -1;
            }

            void main()
            {
                PrintInteger(choose(1));
                PrintInteger(99);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(42, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(99, interpreter.action_snapshots[-1].arg_values[0])

    def test_switch_normal_fallthrough_cleans_local_and_discriminant(self):
        ncs = self.compile(
            """
            void main()
            {
                int result = 0;
                switch (1)
                {
                    case 1:
                        int local = 5;
                        result = local;
                }

                int after = 7;
                PrintInteger(result);
                PrintInteger(after);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(5, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(7, interpreter.action_snapshots[-1].arg_values[0])

    def test_continue_through_switch_targets_enclosing_loop(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 0;
                while (value < 1)
                {
                    int loopLocal = 10;
                    switch (value)
                    {
                        case 0:
                            int switchLocal = 20;
                            value = 1;
                            continue;
                    }

                    value = 99;
                }

                PrintInteger(value);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.action_snapshots[-1].arg_values[0])

    def test_switch_rejects_case_jump_over_local_declaration(self):
        source = """
            void main()
            {
                switch (1)
                {
                    int skipped;
                    case 1:
                        break;
                }
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_switch_with_default(self):
        ncs = self.compile(
            """
            void main()
            {
                switch (4)
                {
                    case 1:
                        PrintInteger(1);
                        break;
                    case 2:
                        PrintInteger(2);
                        break;
                    case 3:
                        PrintInteger(3);
                        break;
                    default:
                        PrintInteger(4);
                        break;
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(4, interpreter.action_snapshots[0].arg_values[0])

    def test_switch_scoped_blocks(self):
        ncs = self.compile(
            """
            void main()
            {
                switch (2)
                {
                    case 1:
                    {
                        int inner = 10;
                        PrintInteger(inner);
                    }
                    break;

                    case 2:
                    {
                        int inner = 20;
                        PrintInteger(inner);
                    }
                    break;
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(20, interpreter.action_snapshots[-1].arg_values[0])

    # endregion

    def test_scope(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 1;

                if (value == 1)
                {
                    value = 2;
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

    def test_scoped_block(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1;

                {
                    int b = 2;
                    PrintInteger(a);
                    PrintInteger(b);
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[-1].arg_values[0])

    # region If/Else Conditions
    def test_if(self):
        ncs = self.compile(
            """
            void main()
            {
                if(0)
                {
                    PrintInteger(0);
                }

                if(1)
                {
                    PrintInteger(1);
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(1, interpreter.action_snapshots[0].arg_values[0])

    def test_if_multiple_conditions(self):
        ncs = self.compile(
            """
            void main()
            {
                if(1 && 2 && 3)
                {
                    PrintInteger(0);
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

    def test_if_else(self):
        ncs = self.compile(
            """
            void main()
            {
                if (0) {    PrintInteger(0); }
                else {      PrintInteger(1); }

                if (1) {    PrintInteger(2); }
                else {      PrintInteger(3); }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(1, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[1].arg_values[0])

    def test_if_else_if(self):
        ncs = self.compile(
            """
            void main()
            {
                if (0)      { PrintInteger(0); }
                else if (0) { PrintInteger(1); }

                if (1)      { PrintInteger(2); } // hit
                else if (1) { PrintInteger(3); }

                if (1)      { PrintInteger(4); } // hit
                else if (0) { PrintInteger(5); }

                if (0)      { PrintInteger(6); }
                else if (1) { PrintInteger(7); } // hit
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(3, len(interpreter.action_snapshots))
        self.assertEqual(2, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(4, interpreter.action_snapshots[1].arg_values[0])
        self.assertEqual(7, interpreter.action_snapshots[2].arg_values[0])

    def test_if_else_if_else(self):
        ncs = self.compile(
            """
            void main()
            {
                if (0)      { PrintInteger(0); }
                else if (0) { PrintInteger(1); }
                else        { PrintInteger(3); } // hit

                if (0)      { PrintInteger(4); }
                else if (1) { PrintInteger(5); } // hit
                else        { PrintInteger(6); }

                if (1)      { PrintInteger(7); } // hit
                else if (1) { PrintInteger(8); }
                else        { PrintInteger(9); }

                if (1)      { PrintInteger(10); } //hit
                else if (0) { PrintInteger(11); }
                else        { PrintInteger(12); }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(4, len(interpreter.action_snapshots))
        self.assertEqual(3, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(5, interpreter.action_snapshots[1].arg_values[0])
        self.assertEqual(7, interpreter.action_snapshots[2].arg_values[0])
        self.assertEqual(10, interpreter.action_snapshots[3].arg_values[0])

    def test_single_statement_if(self):
        ncs = self.compile(
            """
            void main()
            {
                if (1) PrintInteger(222);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(222, interpreter.action_snapshots[-1].arg_values[0])

    def test_single_statement_else_if_else(self):
        ncs = self.compile(
            """
            void main()
            {
                if (0) PrintInteger(11);
                else if (0) PrintInteger(22);
                else PrintInteger(33);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(33, interpreter.action_snapshots[-1].arg_values[0])

    # endregion

    # region While
    def test_while_loop(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 3;
                while (value)
                {
                    PrintInteger(value);
                    value -= 1;
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(3, len(interpreter.action_snapshots))
        self.assertEqual(3, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[1].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[2].arg_values[0])

    def test_while_loop_with_break(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 3;
                while (value)
                {
                    PrintInteger(value);
                    value -= 1;
                    break;
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(3, interpreter.action_snapshots[0].arg_values[0])

    def test_while_loop_with_continue(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 3;
                while (value)
                {
                    PrintInteger(value);
                    value -= 1;
                    continue;
                    PrintInteger(99);
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(3, len(interpreter.action_snapshots))
        self.assertEqual(3, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[1].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[2].arg_values[0])

    def test_nested_loop_does_not_change_later_break_unwind(self):
        ncs = self.compile(
            """
            void main()
            {
                int outer = 0;
                while (outer < 1)
                {
                    while (0)
                    {
                        int deadLocal = 1;
                    }

                    {
                        int liveLocal = 7;
                        outer = 1;
                        break;
                    }
                }

                PrintInteger(outer);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.action_snapshots[-1].arg_values[0])

    def test_while_loop_scope(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 11;
                int outer = 22;
                while (value)
                {
                    int inner = 33;
                    value = 0;
                    continue;
                    outer = 99;
                }

                PrintInteger(outer);
                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(22, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(0, interpreter.action_snapshots[1].arg_values[0])

    # endregion

    # region Do While
    def test_do_while_loop(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 3;
                do
                {
                    PrintInteger(value);
                    value -= 1;
                } while (value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(3, len(interpreter.action_snapshots))
        self.assertEqual(3, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[1].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[2].arg_values[0])

    def test_do_while_loop_with_break(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 3;
                do
                {
                    PrintInteger(value);
                    value -= 1;
                    break;
                } while (value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(3, interpreter.action_snapshots[0].arg_values[0])

    def test_do_while_loop_with_continue(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 3;
                do
                {
                    PrintInteger(value);
                    value -= 1;
                    continue;
                    PrintInteger(99);
                } while (value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(3, len(interpreter.action_snapshots))
        self.assertEqual(3, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[1].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[2].arg_values[0])

    def test_do_while_loop_scope(self):
        ncs = self.compile(
            """
            void main()
            {
                int outer = 11;
                int value = 22;
                do
                {
                    int inner = 33;
                    value = 0;
                } while (value);

                PrintInteger(outer);
                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(11, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(0, interpreter.action_snapshots[1].arg_values[0])

    # endregion

    # region For Loop
    def test_for_loop(self):
        ncs = self.compile(
            """
            void main()
            {
                int i = 99;
                for (i = 1; i <= 3; i += 1)
                {
                    PrintInteger(i);
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(3, len(interpreter.action_snapshots))
        self.assertEqual(1, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[1].arg_values[0])
        self.assertEqual(3, interpreter.action_snapshots[2].arg_values[0])

    def test_for_loop_with_break(self):
        ncs = self.compile(
            """
            void main()
            {
                int i = 99;
                for (i = 1; i <= 3; i += 1)
                {
                    PrintInteger(i);
                    break;
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(1, interpreter.action_snapshots[0].arg_values[0])

    def test_for_loop_with_continue(self):
        ncs = self.compile(
            """
            void main()
            {
                int i = 99;
                for (i = 1; i <= 3; i += 1)
                {
                    PrintInteger(i);
                    continue;
                    PrintInteger(99);
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(3, len(interpreter.action_snapshots))
        self.assertEqual(1, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[1].arg_values[0])
        self.assertEqual(3, interpreter.action_snapshots[2].arg_values[0])

    def test_for_loop_scope(self):
        ncs = self.compile(
            """
            void main()
            {
                int i = 11;
                int outer = 22;
                for (i = 0; i <= 5; i += 1)
                {
                    int inner = 33;
                    break;
                }

                PrintInteger(i);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(0, interpreter.action_snapshots[-1].arg_values[0])

    def test_single_statement_loop_bodies(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 0;
                while (value < 1) value++;
                do value++; while (value < 2);
                for (value = 0; value < 2; value++) PrintInteger(value);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(0, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[1].arg_values[0])

    def test_for_loop_optional_clauses(self):
        ncs = self.compile(
            """
            void main()
            {
                int i = 0;

                for (; i < 2; i++)
                    PrintInteger(i);

                for (i = 0; ; i++)
                {
                    if (i == 2) break;
                }
                PrintInteger(i);

                for (i = 0; i < 2; ) i++;
                PrintInteger(i);

                for (;;) break;
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual([0, 1, 2, 2], [snap.arg_values[0] for snap in interpreter.action_snapshots])

    def test_null_statement_for_body_is_allowed(self):
        self.compile(
            """
            void main()
            {
                for (; 0; );
            }
            """
        )

    def test_null_statement_if_body_is_rejected(self):
        with self.assertRaises(CompileError):
            self.compile("void main() { if (1); }")

    def test_null_statement_else_body_is_rejected(self):
        with self.assertRaises(CompileError):
            self.compile("void main() { if (1) {} else; }")

    def test_null_statement_while_body_is_rejected(self):
        with self.assertRaises(CompileError):
            self.compile("void main() { while (0); }")

    def test_null_statement_do_body_is_rejected(self):
        with self.assertRaises(CompileError):
            self.compile("void main() { do; while (0); }")

    def test_for_loop_declaration_initializer_is_rejected(self):
        with self.assertRaises(CompileError):
            self.compile(
                """
                void main()
                {
                    for (int i = 0; i < 3; i++)
                    {
                    }
                }
                """
            )

    # endregion

    def test_float_notations(self):
        ncs = self.compile(
            """
            void main()
            {
                PrintFloat(1.0f);
                PrintFloat(2.0);
                PrintFloat(3f);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(3, interpreter.action_snapshots[-1].arg_values[0])

    def test_multi_declarations(self):
        ncs = self.compile(
            """
            void main()
            {
                int value1, value2 = 1, value3 = 2;

                PrintInteger(value1);
                PrintInteger(value2);
                PrintInteger(value3);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[-1].arg_values[0])

    def test_local_declarations(self):
        ncs = self.compile(
            """
            void main()
            {
                int INT;
                float FLOAT;
                string STRING;
                location LOCATION;
                effect EFFECT;
                talent TALENT;
                event EVENT;
                vector VECTOR;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

    def test_global_declarations(self):
        ncs = self.compile(
            """
            int INT;
            float FLOAT;
            string STRING;
            location LOCATION;
            effect EFFECT;
            talent TALENT;
            event EVENT;
            vector VECTOR;

            void main()
            {

            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertTrue(any(inst for inst in ncs.instructions if inst.ins_type == NCSInstructionType.SAVEBP))

    def test_global_declarator_lists(self):
        ncs = self.compile(
            """
            int first, second = 2, third;
            const int OFFSET = 3, TOTAL = OFFSET + 4;

            void main()
            {
                PrintInteger(first);
                PrintInteger(second);
                PrintInteger(third);
                PrintInteger(TOTAL);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual([0, 2, 0, 7], [snap.arg_values[0] for snap in interpreter.action_snapshots])

    def test_global_initializations(self):
        ncs = self.compile(
            """
            int INT = 0;
            float FLOAT = 0.0;
            string STRING = "";
            vector VECTOR = [0.0, 0.0, 0.0];

            void main()
            {
                PrintInteger(INT);
                PrintFloat(FLOAT);
                PrintString(STRING);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(interpreter.action_snapshots[-3].arg_values[0], 0)
        self.assertEqual(interpreter.action_snapshots[-2].arg_values[0], 0.0)
        self.assertEqual(interpreter.action_snapshots[-1].arg_values[0], "")
        self.assertTrue(any(inst for inst in ncs.instructions if inst.ins_type == NCSInstructionType.SAVEBP))

    def test_global_initializer_can_call_engine_function(self):
        ncs = self.compile(
            """
            object FIRST_PLAYER = GetFirstPC();

            void main()
            {
            }
            """
        )

        self.assertTrue(
            any(
                instruction.ins_type == NCSInstructionType.ACTION
                and instruction.args[0]
                == next(
                    i
                    for i, function in enumerate(KOTOR_FUNCTIONS)
                    if function.name == "GetFirstPC"
                )
                for instruction in ncs.instructions
            )
        )

    def test_global_initializer_can_call_registered_user_function(self):
        ncs = self.compile(
            """
            int ComputeInitialValue();
            int VALUE = ComputeInitialValue();

            int ComputeInitialValue()
            {
                return 42;
            }

            void main()
            {
                PrintInteger(VALUE);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(42, interpreter.action_snapshots[-1].arg_values[0])

    def test_global_initializer_can_call_function_from_include(self):
        helper = """
            int IncludedInitialValue()
            {
                return 91;
            }
        """.encode(encoding="windows-1252")

        ncs = self.compile(
            """
            #include "initializer_helper"

            int VALUE = IncludedInitialValue();

            void main()
            {
                PrintInteger(VALUE);
            }
            """,
            library={"initializer_helper": helper},
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(91, interpreter.action_snapshots[-1].arg_values[0])

    def test_global_initializer_cannot_call_later_function_without_prototype(self):
        source = """
            int VALUE = ComputeInitialValue();

            int ComputeInitialValue()
            {
                return 73;
            }

            void main()
            {
                PrintInteger(VALUE);
            }
        """

        with self.assertRaisesRegex(CompileError, "Undefined function 'ComputeInitialValue'"):
            self.compile(source)

    def test_global_initializer_cannot_see_later_compile_time_constant(self):
        source = """
            int VALUE = LATER_VALUE;
            const int LATER_VALUE = 5;

            void main()
            {
            }
        """

        with self.assertRaisesRegex(CompileError, "Undefined variable 'LATER_VALUE'"):
            self.compile(source)

    def test_global_initializer_can_see_earlier_compile_time_constant(self):
        ncs = self.compile(
            """
            const int EARLIER_VALUE = 5;
            int VALUE = EARLIER_VALUE;

            void main()
            {
                PrintInteger(VALUE);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(5, interpreter.action_snapshots[-1].arg_values[0])

    def test_function_call_requires_prior_declaration(self):
        source = """
            void main()
            {
                Later();
            }

            void Later()
            {
            }
        """

        with self.assertRaisesRegex(CompileError, "Undefined function 'Later'"):
            self.compile(source)

    def test_function_call_accepts_prior_prototype(self):
        ncs = self.compile(
            """
            int Later();

            void main()
            {
                PrintInteger(Later());
            }

            int Later()
            {
                return 73;
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(73, interpreter.action_snapshots[-1].arg_values[0])

    def test_recursive_function_is_visible_in_its_own_body(self):
        self.compile(
            """
            int CountDown(int value)
            {
                if (value == 0)
                {
                    return 0;
                }
                return CountDown(value - 1);
            }

            void main()
            {
                CountDown(2);
            }
            """
        )

    def test_function_before_include_cannot_see_later_included_function(self):
        source = """
            void Helper()
            {
                IncludedLater();
            }

            #include "later_helper"

            void main()
            {
                Helper();
            }
        """
        library = {"later_helper": b"void IncludedLater() {}"}

        with self.assertRaisesRegex(CompileError, "Undefined function 'IncludedLater'"):
            self.compile(source, library=library)

    def test_function_after_include_can_see_included_function(self):
        source = """
            #include "earlier_helper"

            void main()
            {
                IncludedEarlier();
            }
        """
        library = {"earlier_helper": b"void IncludedEarlier() {}"}

        self.compile(source, library=library)

    def test_duplicate_global_variables_are_rejected_during_registration(self):
        source = """
            int VALUE;
            float VALUE;

            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_duplicate_function_parameters_are_rejected_during_registration(self):
        source = """
            void helper(int value, float value) {}
            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_duplicate_struct_names_are_rejected_during_registration(self):
        source = """
            struct Pair
            {
                int first;
            };

            struct Pair
            {
                int second;
            };

            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_duplicate_struct_members_are_rejected_during_registration(self):
        source = """
            struct Pair
            {
                int value;
                float value;
            };

            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_unknown_struct_member_type_is_rejected_during_registration(self):
        source = """
            struct Wrapper
            {
                struct Missing value;
            };

            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_recursive_struct_layout_is_rejected_during_registration(self):
        source = """
            struct Node
            {
                struct Node next;
            };

            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_struct_storage_supports_target_engine_and_vector_members(self):
        ncs = self.compile(
            """
            struct Payload
            {
                vector direction;
                effect effectValue;
                event eventValue;
                location locationValue;
                talent talentValue;
            };

            void main()
            {
                struct Payload value;
            }
            """
        )

        emitted = [instruction.ins_type for instruction in ncs.instructions]
        self.assertGreaterEqual(emitted.count(NCSInstructionType.RSADDF), 3)
        self.assertIn(NCSInstructionType.RSADDEFF, emitted)
        self.assertIn(NCSInstructionType.RSADDEVT, emitted)
        self.assertIn(NCSInstructionType.RSADDLOC, emitted)
        self.assertIn(NCSInstructionType.RSADDTAL, emitted)

    def test_global_initialization_with_unary(self):
        ncs = self.compile(
            """
            int INT = -1;

            void main()
            {
                PrintInteger(INT);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(interpreter.action_snapshots[-1].arg_values[0], -1)

    def test_comment(self):
        ncs = self.compile(
            """
            void main()
            {
                // int a = "abc"; // [] /*
                int a = 0;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

    def test_multiline_comment(self):
        ncs = self.compile(
            """
            void main()
            {
                /* int
                abc =
                ;; 123
                */

                string aaa = "";
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

    def test_return(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1;

                if (a == 1)
                {
                    PrintInteger(a);
                    return;
                }

                PrintInteger(0);
                return;
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(1, interpreter.action_snapshots[0].arg_values[0])

    def test_return_parenthesis(self):
        ncs = self.compile(
            """
            int test()
            {
                return(321);
            }

            void main()
            {
                int value = test();
                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(321, interpreter.action_snapshots[0].arg_values[0])

    def test_return_parenthesis_constant(self):
        ncs = self.compile(
            """
            int test()
            {
                return(TRUE);
            }

            void main()
            {
                int value = test();
                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.action_snapshots[0].arg_values[0])

    def test_void_function_rejects_return_value(self):
        source = """
            void main()
            {
                return 1;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_nonvoid_function_rejects_bare_return(self):
        source = """
            int test()
            {
                return;
            }

            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_nonvoid_function_rejects_wrong_return_type(self):
        source = """
            int test()
            {
                return "wrong";
            }

            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_nonvoid_function_requires_return_on_all_paths(self):
        source = """
            int test(int value)
            {
                if (value)
                {
                    return 1;
                }
            }

            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_nonvoid_function_accepts_complete_if_else_returns(self):
        ncs = self.compile(
            """
            int test(int value)
            {
                if (value)
                {
                    return 10;
                }
                else
                {
                    return 20;
                }
            }

            void main()
            {
                PrintInteger(test(1));
                PrintInteger(test(0));
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(10, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(20, interpreter.action_snapshots[-1].arg_values[0])

    def test_nonvoid_switch_alone_does_not_prove_all_paths_return(self):
        source = """
            int test(int value)
            {
                switch (value)
                {
                    case 0:
                        return 10;
                    default:
                        return 20;
                }
            }

            void main() {}
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_vector_return_copies_full_value(self):
        ncs = self.compile(
            """
            vector make_vector()
            {
                return [1.0, 2.0, 3.0];
            }

            void main()
            {
                vector value = make_vector();
            }
            """
        )

        self.assertTrue(
            any(
                instruction.ins_type == NCSInstructionType.CPDOWNSP
                and len(instruction.args) >= 2
                and instruction.args[1] == 12
                for instruction in ncs.instructions
            )
        )

    def test_struct_return_copies_full_value(self):
        ncs = self.compile(
            """
            struct Pair
            {
                int first;
                int second;
            };

            struct Pair make_pair()
            {
                struct Pair value;
                value.first = 1;
                value.second = 2;
                return value;
            }

            void main()
            {
                struct Pair value = make_pair();
            }
            """
        )

        self.assertTrue(
            any(
                instruction.ins_type == NCSInstructionType.CPDOWNSP
                and len(instruction.args) >= 2
                and instruction.args[1] == 8
                for instruction in ncs.instructions
            )
        )

    def test_int_parenthesis_declaration(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = (123);
                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(123, interpreter.action_snapshots[-1].arg_values[0])

    def test_include_builtin(self):
        otherscript = """
            void TestFunc()
            {
                PrintInteger(123);
            }
        """.encode(encoding="windows-1252")

        ncs = self.compile(
            """
            #include "otherscript"

            void main()
            {
                TestFunc();
            }
        """,
            library={"otherscript": otherscript},
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

    def test_include_lookup(self):
        includetest_script_path = Path("./tests/files").resolve()
        if not includetest_script_path.is_dir():
            import errno
            msg = "Could not find includetest.nss in the include folder!"
            raise FileNotFoundError(errno.ENOENT, msg, str(includetest_script_path))
        ncs = self.compile(
            """
            #include "includetest"

            void main()
            {
                TestFunc();
            }
        """,
            library_lookup=includetest_script_path,
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

    def test_nested_include(self):
        first_script = """
            int SOME_COST = 13;

            void TestFunc(int value)
            {
                PrintInteger(value);
            }
        """.encode(encoding="windows-1252")

        second_script = """
            #include "first_script"
        """.encode(encoding="windows-1252")

        ncs = self.compile(
            """
            #include "second_script"

            void main()
            {
                TestFunc(SOME_COST);
            }
        """,
            library={"first_script": first_script, "second_script": second_script},
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(13, interpreter.action_snapshots[0].arg_values[0])

    def test_include_once_is_case_insensitive(self):
        shared_script = b"int SHARED_VALUE = 17;"

        ncs = self.compile(
            """
            #include "Shared"
            #include "sHaReD"

            void main()
            {
                PrintInteger(SHARED_VALUE);
            }
            """,
            library={"shared": shared_script},
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(17, interpreter.action_snapshots[-1].arg_values[0])

    def test_recursive_include_is_rejected_case_insensitively(self):
        library = {
            "alpha": b'#include "Beta"',
            "beta": b'#include "ALPHA"',
        }
        script = '#include "alpha"\nvoid main() {}'

        with self.assertRaisesRegex(CompileError, "Recursive include detected"):
            self.compile(script, library=library)

    def test_include_depth_limit_counts_root_as_first_level(self):
        library = {
            "level_a": b'#include "level_b"',
            "level_b": b'#include "level_c"',
            "level_c": b"",
        }
        script = '#include "level_a"\nvoid main() {}'

        with self.assertRaisesRegex(CompileError, "Maximum include depth"):
            self.compile(script, library=library, max_include_depth=3)

    def test_default_include_depth_matches_bioware(self):
        self.assertEqual(16, DEFAULT_MAX_INCLUDE_DEPTH)

        allowed_library: dict[str, bytes] = {}
        for index in range(15):
            next_name = f"level_{index + 1}"
            allowed_library[f"level_{index}"] = (
                f'#include "{next_name}"'.encode() if index < 14 else b""
            )
        self.compile('#include "level_0"\nvoid main() {}', library=allowed_library)

        too_deep_library = dict(allowed_library)
        too_deep_library["level_14"] = b'#include "level_15"'
        too_deep_library["level_15"] = b""
        with self.assertRaisesRegex(CompileError, "Maximum include depth"):
            self.compile('#include "level_0"\nvoid main() {}', library=too_deep_library)

    def test_missing_include(self):
        source = """
            #include "otherscript"

            void main()
            {
                TestFunc();
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_global_int_addition_assignment(self):
        ncs = self.compile(
            """
            int global1 = 1;
            int global2 = 2;

            void main()
            {
                int local1 = 3;
                int local2 = 4;

                global1 += local1;
                global2 = local2 + global1;

                PrintInteger(global1);
                PrintInteger(global2);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(4, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(8, interpreter.action_snapshots[-1].arg_values[0])

    def test_global_int_subtraction_assignment(self):
        ncs = self.compile(
            """
            int global1 = 1;
            int global2 = 10;

            void main()
            {
                int local1 = 100;
                int local2 = 1000;

                global1 -= local1;              // 1 - 100 = -99
                global2 = local2 - global1;     // 1000 - -99 = 1099

                PrintInteger(global1);
                PrintInteger(global2);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(-99, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(1099, interpreter.action_snapshots[-1].arg_values[0])

    def test_global_int_multiplication_assignment(self):
        ncs = self.compile(
            """
            int global1 = 1;
            int global2 = 10;

            void main()
            {
                int local1 = 100;
                int local2 = 1000;

                global1 *= local1;              // 1 * 100 = 100
                global2 = local2 * global1;     // 1000 * 100 = 100000

                PrintInteger(global1);
                PrintInteger(global2);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(100, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(100000, interpreter.action_snapshots[-1].arg_values[0])

    def test_global_int_division_assignment(self):
        ncs = self.compile(
            """
            int global1 = 1000;
            int global2 = 100;

            void main()
            {
                int local1 = 10;
                int local2 = 1;

                global1 /= local1;              // 1000 / 10 = 100
                global2 = global1 / local2;     // 100 / 1 = 100

                PrintInteger(global1);
                PrintInteger(global2);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(100, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(100, interpreter.action_snapshots[-1].arg_values[0])

    def test_imported_global_variable(self):
        otherscript = """
            int iExperience = 55;
        """.encode(encoding="windows-1252")

        ncs = self.compile(
            """
            #include "otherscript"

            void main()
            {
                object oPlayer = GetPCSpeaker();
                GiveXPToCreature(oPlayer, iExperience);
            }
        """,
            library={"otherscript": otherscript},
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(55, interpreter.action_snapshots[1].arg_values[1])

    def test_declaration_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a;
                PrintInteger(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-1].arg_values[0])

    def test_declaration_float(self):
        ncs = self.compile(
            """
            void main()
            {
                float a;
                PrintFloat(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0.0, interpreter.action_snapshots[-1].arg_values[0])

    def test_declaration_string(self):
        ncs = self.compile(
            """
            void main()
            {
                string a;
                PrintString(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual("", interpreter.action_snapshots[-1].arg_values[0])

    def test_vector(self):
        ncs = self.compile(
            """
            void main()
            {
                vector vec = Vector(2.0, 4.0, 4.0);
                float mag = VectorMagnitude(vec);
                PrintFloat(mag);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.set_mock("Vector", Vector3)
        interpreter.set_mock("VectorMagnitude", lambda vec: vec.magnitude())
        interpreter.run()

        self.assertEqual(6.0, interpreter.action_snapshots[-1].arg_values[0])

    def test_vector_notation(self):
        ncs = self.compile(
            """
            void main()
            {
                vector vec = [1.0, 2.0, 3.0];
                PrintFloat(vec.x);
                PrintFloat(vec.y);
                PrintFloat(vec.z);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1.0, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(2.0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(3.0, interpreter.action_snapshots[-1].arg_values[0])

    def test_vector_abbreviated_notation_zero_fills_trailing_components(self):
        ncs = self.compile(
            """
            void main()
            {
                vector empty = [];
                vector one = [1.0];
                vector two = [2.0, 3.0];

                PrintFloat(empty.x);
                PrintFloat(empty.y);
                PrintFloat(empty.z);
                PrintFloat(one.x);
                PrintFloat(one.y);
                PrintFloat(one.z);
                PrintFloat(two.x);
                PrintFloat(two.y);
                PrintFloat(two.z);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 0.0],
            [snap.arg_values[0] for snap in interpreter.action_snapshots],
        )

    def test_vector_get_components(self):
        ncs = self.compile(
            """
            void main()
            {
                vector vec = Vector(2.0, 4.0, 6.0);
                PrintFloat(vec.x);
                PrintFloat(vec.y);
                PrintFloat(vec.z);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.set_mock("Vector", Vector3)
        interpreter.run()

        self.assertEqual(2.0, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(4.0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(6.0, interpreter.action_snapshots[-1].arg_values[0])

    def test_vector_set_components(self):
        ncs = self.compile(
            """
            void main()
            {
                vector vec = Vector(0.0, 0.0, 0.0);
                vec.x = 2.0;
                vec.y = 4.0;
                vec.z = 6.0;
                PrintFloat(vec.x);
                PrintFloat(vec.y);
                PrintFloat(vec.z);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.set_mock("Vector", Vector3)
        interpreter.run()

        self.assertEqual(2.0, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(4.0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(6.0, interpreter.action_snapshots[-1].arg_values[0])

    def test_struct_member_declarator_lists(self):
        ncs = self.compile(
            """
            struct Pair
            {
                int left, right;
                float x, y;
            };

            void main()
            {
                struct Pair value;
                value.left = 11;
                value.right = 22;
                PrintInteger(value.left);
                PrintInteger(value.right);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual([11, 22], [snap.arg_values[0] for snap in interpreter.action_snapshots])

    def test_bare_struct_type_name_is_rejected(self):
        with self.assertRaises(CompileError):
            self.compile(
                """
                struct Pair
                {
                    int value;
                };

                Pair value;
                void main()
                {
                }
                """
            )

    def test_declaration_type_categories_reject_void_action_and_itemproperty(self):
        scripts = [
            "void main() { void value; }",
            "void helper(void value) {} void main() {}",
            "struct Bad { void value; }; void main() {}",
            "void main() { action value; }",
            "void main() { itemproperty value; }",
        ]
        for script in scripts:
            with self.subTest(script=script), self.assertRaises(CompileError):
                self.compile(script)

    def test_struct_get_members(self):
        ncs = self.compile(
            """
            struct ABC
            {
                int value1;
                string value2;
                float value3;
            };

            void main()
            {
                struct ABC abc;
                PrintInteger(abc.value1);
                PrintString(abc.value2);
                PrintFloat(abc.value3);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual("", interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(0.0, interpreter.action_snapshots[-1].arg_values[0])

    def test_struct_get_invalid_member(self):
        source = """
            struct ABC
            {
                int value1;
                string value2;
                float value3;
            };

            void main()
            {
                struct ABC abc;
                PrintFloat(abc.value4);
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_struct_set_members(self):
        ncs = self.compile(
            """
            struct ABC
            {
                int value1;
                string value2;
                float value3;
            };

            void main()
            {
                struct ABC abc;
                abc.value1 = 123;
                abc.value2 = "abc";
                abc.value3 = 3.14;
                PrintInteger(abc.value1);
                PrintString(abc.value2);
                PrintFloat(abc.value3);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(123, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual("abc", interpreter.action_snapshots[-2].arg_values[0])
        self.assertAlmostEqual(3.14, interpreter.action_snapshots[-1].arg_values[0].value)

    def test_prefix_increment_sp_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1;
                int b = ++a;

                PrintInteger(a);
                PrintInteger(b);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[-1].arg_values[0])

    def test_prefix_increment_bp_int(self):
        ncs = self.compile(
            """
            int a = 1;

            void main()
            {
                int b = ++a;

                PrintInteger(a);
                PrintInteger(b);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[-1].arg_values[0])

    def test_postfix_increment_sp_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1;
                int b = a++;

                PrintInteger(a);
                PrintInteger(b);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-1].arg_values[0])

    def test_postfix_increment_bp_int(self):
        ncs = self.compile(
            """
            int a = 1;

            void main()
            {
                int b = a++;

                PrintInteger(a);
                PrintInteger(b);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-1].arg_values[0])

    def test_prefix_decrement_sp_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1;
                int b = --a;

                PrintInteger(a);
                PrintInteger(b);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(0, interpreter.action_snapshots[-1].arg_values[0])

    def test_prefix_decrement_bp_int(self):
        ncs = self.compile(
            """
            int a = 1;

            void main()
            {
                int b = --a;

                PrintInteger(a);
                PrintInteger(b);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(0, interpreter.action_snapshots[-1].arg_values[0])

    def test_postfix_decrement_sp_int(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 1;
                int b = a--;

                PrintInteger(a);
                PrintInteger(b);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-1].arg_values[0])

    def test_postfix_decrement_bp_int(self):
        ncs = self.compile(
            """
            int a = 1;

            void main()
            {
                int b = a--;

                PrintInteger(a);
                PrintInteger(b);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(0, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(1, interpreter.action_snapshots[-1].arg_values[0])

    def test_assignmentless_expression(self):
        ncs = self.compile(
            """
            void main()
            {
                int a = 123;

                1;
                GetCheatCode(1);
                "abc";

                PrintInteger(a);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(123, interpreter.action_snapshots[-1].arg_values[0])

    def test_function_cannot_see_later_compile_time_constant(self):
        source = """
            void main()
            {
                int value = LATER_VALUE;
            }

            const int LATER_VALUE = 5;
        """

        with self.assertRaisesRegex(CompileError, "Undefined variable 'LATER_VALUE'"):
            self.compile(source)

    def test_function_can_see_earlier_compile_time_constant(self):
        ncs = self.compile(
            """
            const int EARLIER_VALUE = 5;

            void main()
            {
                PrintInteger(EARLIER_VALUE);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(5, interpreter.action_snapshots[-1].arg_values[0])

    def test_default_parameter_cannot_see_later_compile_time_constant(self):
        source = """
            void Helper(int value = LATER_VALUE);
            const int LATER_VALUE = 5;

            void Helper(int value)
            {
            }

            void main()
            {
                Helper();
            }
        """

        with self.assertRaisesRegex(CompileError, "Non-constant default value"):
            self.compile(source)

    def test_default_parameter_can_see_earlier_compile_time_constant(self):
        self.compile(
            """
            const int EARLIER_VALUE = 5;
            void Helper(int value = EARLIER_VALUE);

            void Helper(int value)
            {
            }

            void main()
            {
                Helper();
            }
            """
        )

    # region Script Subroutines
    def test_prototype_no_args(self):
        ncs = self.compile(
            """
            void test();

            void main()
            {
                test();
            }

            void test()
            {
                PrintInteger(56);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(56, interpreter.action_snapshots[0].arg_values[0])

    def test_prototype_with_arg(self):
        ncs = self.compile(
            """
            void test(int value);

            void main()
            {
                test(57);
            }

            void test(int value)
            {
                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(57, interpreter.action_snapshots[0].arg_values[0])

    def test_prototype_with_three_args(self):
        ncs = self.compile(
            """
            void test(int a, int b, int c)
            {
                PrintInteger(a);
                PrintInteger(b);
                PrintInteger(c);
            }

            void main()
            {
                int a = 1, b = 2, c = 3;
                test(a, b, c);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(3, interpreter.action_snapshots[-1].arg_values[0])

    def test_prototype_with_many_args(self):
        ncs = self.compile(
            """
            void test(int a, effect z, int b, int c, int d = 4)
            {
                PrintInteger(a);
                PrintInteger(b);
                PrintInteger(c);
                PrintInteger(d);
            }

            void main()
            {
                int a = 1, b = 2, c = 3;
                effect z;

                test(a, z, b, c);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, interpreter.action_snapshots[-4].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[-3].arg_values[0])
        self.assertEqual(3, interpreter.action_snapshots[-2].arg_values[0])
        self.assertEqual(4, interpreter.action_snapshots[-1].arg_values[0])

    def test_prototype_with_default_arg(self):
        ncs = self.compile(
            """
            void test(int value = 57);

            void main()
            {
                test();
            }

            void test(int value = 57)
            {
                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(57, interpreter.action_snapshots[0].arg_values[0])

    def test_prototype_with_default_constant_arg(self):
        ncs = self.compile(
            """
            void test(int value = DAMAGE_TYPE_COLD);

            void main()
            {
                test();
            }

            void test(int value = DAMAGE_TYPE_COLD)
            {
                PrintInteger(value);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(32, interpreter.action_snapshots[0].arg_values[0])

    def test_prototype_missing_arg(self):
        source = """
            void test(int value);

            void main()
            {
                test();
            }

            void test(int value)
            {
                PrintInteger(value);
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_prototype_missing_arg_and_default(self):
        source = """
            void test(int value1, int value2 = 123);

            void main()
            {
                test();
            }

            void test(int value1, int value2 = 123)
            {
                PrintInteger(value1);
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_prototype_default_before_required(self):
        source = """
            void test(int value1 = 123, int value2);

            void main()
            {
                test(123, 123);
            }

            void test(int value1 = 123, int value2)
            {
                PrintInteger(value1);
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_redefine_function(self):
        script = """
            void test()
            {

            }

            void test()
            {

            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_identical_repeated_prototypes_are_accepted(self):
        ncs = self.compile(
            """
            void test(int value);
            void test(int renamed);

            void test(int value)
            {
                PrintInteger(value);
            }

            void main()
            {
                test(61);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(61, interpreter.action_snapshots[-1].arg_values[0])

    def test_identical_prototype_after_definition_is_accepted(self):
        ncs = self.compile(
            """
            void test()
            {
                PrintInteger(62);
            }

            void test();

            void main()
            {
                test();
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(62, interpreter.action_snapshots[-1].arg_values[0])

    def test_conflicting_repeated_prototypes_are_rejected(self):
        script = """
            void test(int value);
            void test(float value);
            void main() {}
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_unused_prototype_without_definition_is_accepted(self):
        ncs = self.compile(
            """
            void unused_helper(int value);

            void main()
            {
                PrintInteger(63);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(63, interpreter.action_snapshots[-1].arg_values[0])

    def test_called_prototype_without_definition_is_rejected(self):
        script = """
            void missing_helper(int value);

            void main()
            {
                missing_helper(64);
            }
        """

        with self.assertRaisesRegex(CompileError, "missing_helper"):
            self.compile(script)

    def test_prototype_does_not_emit_executable_stub(self):
        ncs = self.compile(
            """
            void unused_helper();
            void unused_helper();
            void main() {}
            """
        )

        # Only the actual main implementation owns a function-entry NOP.
        self.assertEqual(1, sum(i.ins_type == NCSInstructionType.NOP for i in ncs.instructions))

    def test_prototype_and_definition_param_mismatch(self):
        script = """
            void test(int a);

            void test()
            {

            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_prototype_and_definition_default_param_mismatch(self):
        """This test is disabled for now."""
        # script = """
        #     void test(int a = 1);
        #
        #     void test(int a = 2)
        #     {
        #
        #     }
        # """
        # self.assertRaises(CompileError, self.compile, script)

    def test_prototype_and_definition_return_mismatch(self):
        script = """
            void test(int a);

            int test(int a)
            {

            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_main_must_return_void(self):
        source = """
            int main()
            {
                return 1;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_main_must_have_no_parameters(self):
        source = """
            void main(int value)
            {
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_starting_conditional_must_return_int(self):
        source = """
            void StartingConditional()
            {
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_starting_conditional_must_have_no_parameters(self):
        source = """
            int StartingConditional(int value)
            {
                return value;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_valid_starting_conditional_entry_point(self):
        ncs = self.compile(
            """
            int StartingConditional()
            {
                return 1;
            }
            """
        )

        self.assertEqual(NCSInstructionType.RSADDI, ncs.instructions[0].ins_type)
        self.assertEqual(NCSInstructionType.JSR, ncs.instructions[1].ins_type)
        self.assertEqual(NCSInstructionType.RETN, ncs.instructions[2].ins_type)

    def test_invalid_main_takes_precedence_over_starting_conditional(self):
        source = """
            int main()
            {
                return 1;
            }

            int StartingConditional()
            {
                return 1;
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_root_main_prototype_selects_included_main_implementation(self):
        source = """
            void main();
            #include "main_impl"
        """
        library = {"main_impl": b"void main() { PrintInteger(41); }"}

        ncs = self.compile(source, library=library)
        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(41, interpreter.action_snapshots[-1].arg_values[0])

    def test_root_main_prototype_takes_precedence_over_starting_conditional(self):
        source = """
            int main();

            int StartingConditional()
            {
                return 1;
            }
        """

        with self.assertRaisesRegex(CompileError, "Function 'main' must return void"):
            self.compile(source)

    def test_root_main_prototype_without_implementation_does_not_fall_back(self):
        source = """
            void main();

            int StartingConditional()
            {
                return 1;
            }
        """

        with self.assertRaisesRegex(CompileError, "has no implementation"):
            self.compile(source)

    def test_include_only_main_does_not_override_root_starting_conditional(self):
        source = """
            #include "include_main"

            int StartingConditional()
            {
                return 1;
            }
        """
        library = {"include_main": b"void main() {}"}

        ncs = self.compile(source, library=library)
        self.assertEqual(NCSInstructionType.RSADDI, ncs.instructions[0].ins_type)
        self.assertEqual(NCSInstructionType.JSR, ncs.instructions[1].ins_type)

    def test_call_undefined(self):
        script = """
            void main()
            {
                test(0);
            }
        """

        self.assertRaises(CompileError, self.compile, script)

    def test_call_void_with_no_args(self):
        ncs = self.compile(
            """
            void test()
            {
                PrintInteger(123);
            }

            void main()
            {
                test();
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(123, interpreter.action_snapshots[0].arg_values[0])

    def test_call_void_with_one_arg(self):
        ncs = self.compile(
            """
            void test(int value)
            {
                PrintInteger(value);
            }

            void main()
            {
                test(123);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(123, interpreter.action_snapshots[0].arg_values[0])

    def test_call_void_with_two_args(self):
        ncs = self.compile(
            """
            void test(int value1, int value2)
            {
                PrintInteger(value1);
                PrintInteger(value2);
            }

            void main()
            {
                test(1, 2);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(2, len(interpreter.action_snapshots))
        self.assertEqual(1, interpreter.action_snapshots[0].arg_values[0])
        self.assertEqual(2, interpreter.action_snapshots[1].arg_values[0])

    def test_user_call_rejects_too_many_arguments(self):
        source = """
            void test(int value)
            {
            }

            void main()
            {
                test(1, 2);
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_user_call_rejects_too_few_arguments(self):
        source = """
            void test(int first, int second)
            {
            }

            void main()
            {
                test(1);
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    def test_user_call_accepts_omitted_default_arguments(self):
        ncs = self.compile(
            """
            void test(int first, int second = 7)
            {
                PrintInteger(first + second);
            }

            void main()
            {
                test(5);
            }
            """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual(12, interpreter.action_snapshots[-1].arg_values[0])

    def test_call_int_with_no_args(self):
        ncs = self.compile(
            """
            int test()
            {
                return 5;
            }

            void main()
            {
                int x = test();
                PrintInteger(x);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(5, interpreter.action_snapshots[0].arg_values[0])

    def test_call_int_with_no_args_and_forward_declared(self):
        ncs = self.compile(
            """
            int test();

            int test()
            {
                return 5;
            }

            void main()
            {
                int x = test();
                PrintInteger(x);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual(1, len(interpreter.action_snapshots))
        self.assertEqual(5, interpreter.action_snapshots[0].arg_values[0])

    def test_call_param_mismatch(self):
        source = """
            int test(int a)
            {
                return a;
            }

            void main()
            {
                test("123");
            }
        """

        self.assertRaises(CompileError, self.compile, source)

    # endregion

    # region Local semantic analysis
    def test_duplicate_local_in_same_scope_is_rejected(self):
        script = """
            void main()
            {
                int value;
                int value;
            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_duplicate_local_in_declarator_list_is_rejected(self):
        script = """
            void main()
            {
                int value, value;
            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_function_body_local_may_shadow_parameter(self):
        ncs = self.compile(
            """
            void helper(int value)
            {
                int value = 2;
                PrintInteger(value);
            }

            void main()
            {
                helper(1);
            }
            """
        )
        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual([2], interpreter.action_snapshots[-1].arg_values)

    def test_nested_scope_may_shadow_outer_local(self):
        ncs = self.compile(
            """
            void main()
            {
                int value = 1;
                {
                    int value = 2;
                    PrintInteger(value);
                }
                PrintInteger(value);
            }
            """
        )
        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual([2], interpreter.action_snapshots[0].arg_values)
        self.assertEqual([1], interpreter.action_snapshots[1].arg_values)

    def test_sibling_scopes_may_reuse_local_name(self):
        self.compile(
            """
            void main()
            {
                if (1)
                {
                    int value = 1;
                }
                else
                {
                    int value = 2;
                }
            }
            """
        )

    def test_declarator_initializer_can_see_previous_declarator(self):
        ncs = self.compile(
            """
            void main()
            {
                int first = 7, second = first;
                PrintInteger(second);
            }
            """
        )
        interpreter = Interpreter(ncs)
        interpreter.run()
        self.assertEqual([7], interpreter.action_snapshots[-1].arg_values)

    def test_unreachable_undefined_name_is_rejected(self):
        script = """
            void main()
            {
                return;
                missing = 1;
            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_unreachable_type_error_is_rejected(self):
        script = """
            void main()
            {
                return;
                int value = "wrong";
            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_unreachable_break_is_rejected(self):
        script = """
            void main()
            {
                return;
                break;
            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_unreachable_continue_is_rejected(self):
        script = """
            void main()
            {
                return;
                continue;
            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_constant_dead_branch_is_still_semantically_validated(self):
        script = """
            void main()
            {
                if (0)
                {
                    int value = "wrong";
                }
            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_local_semantic_failure_occurs_before_bytecode_emission(self):
        lexer = NssLexer()
        parser = NssParser(
            library={},
            constants=KOTOR_CONSTANTS,
            functions=KOTOR_FUNCTIONS,
        ).parser
        root = parser.parse(
            """
            int global_value = 7;

            void main()
            {
                int duplicate;
                int duplicate;
            }
            """,
            lexer=lexer.lexer,
            tracking=True,
        )
        ncs = NCS()
        with self.assertRaises(CompileError):
            root.compile(ncs)
        self.assertEqual([], ncs.instructions)

    def test_function_parameter_limit_allows_32(self):
        parameters = ", ".join(
            f"int parameter_{index}" for index in range(MAX_FUNCTION_PARAMETERS)
        )
        self.compile(f"void Helper({parameters}) {{}} void main() {{}}")

    def test_function_parameter_limit_rejects_33(self):
        parameters = ", ".join(
            f"int parameter_{index}" for index in range(MAX_FUNCTION_PARAMETERS + 1)
        )
        with self.assertRaisesRegex(CompileError, "too many parameters"):
            self.compile(f"void Helper({parameters}) {{}} void main() {{}}")

    def test_line_comment_at_eof_without_newline(self):
        self.compile("void main() {}\n// final comment without newline")

    def test_unexpected_eof_is_compile_error(self):
        with self.assertRaisesRegex(CompileError, "unexpected end of file"):
            self.compile("void main() {")

    # endregion

    def test_switch_duplicate_case_values_use_signed_32bit_semantics(self):
        script = """
            void main()
            {
                switch (1)
                {
                    case -1:
                        break;
                    case 0xFFFFFFFF:
                        break;
                }
            }
        """
        self.assertRaises(CompileError, self.compile, script)

    def test_switch_scope_a(self):
        ncs = self.compile(
            """
            int shape;
            int harmful;

            void main()
            {
                object oTarget = OBJECT_SELF;
                effect e1, e2;
                effect e3;

                shape = SHAPE_SPHERE;

                switch (1)
                {
                    case 1:
                        harmful = FALSE;
                        e1 = EffectMovementSpeedIncrease(99);

                        if (1 == 1)
                        {
                            e1 = EffectLinkEffects(e1, EffectVisualEffect(VFX_DUR_SPEED));
                        }

                        GiveXPToCreature(OBJECT_SELF, 100);
                        GetHasSpellEffect(FORCE_POWER_SPEED_BURST, oTarget);
                    break;
                }
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()

        self.assertEqual([8, 0], interpreter.action_snapshots[-1].arg_values)

    def test_switch_scope_b(self):
        ncs = self.compile(
            """
            void test(int abc)
            {
             GiveXPToCreature(GetFirstPC(), abc);
            }

            void main()
            {
                test(123);
            }
        """
        )
        ncs = self.compile(
            """
            void Cort_XP(int abc)
            {
                GiveXPToCreature(GetFirstPC(), abc);
            }

            void main() {
                int abc = 2500;
                Cort_XP(abc);
            }
        """
        )

        interpreter = Interpreter(ncs)
        interpreter.run()


if __name__ == "__main__":
    unittest.main()
