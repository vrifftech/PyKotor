from __future__ import annotations

import pathlib
import sys
import unittest

THIS_SCRIPT_PATH = pathlib.Path(__file__).resolve()
PYKOTOR_PATH = THIS_SCRIPT_PATH.parents[3].resolve()
UTILITY_PATH = THIS_SCRIPT_PATH.parents[5].joinpath("Utility", "src").resolve()


def add_sys_path(path: pathlib.Path) -> None:
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.append(path_string)


if PYKOTOR_PATH.joinpath("pykotor").exists():
    add_sys_path(PYKOTOR_PATH)
if UTILITY_PATH.joinpath("utility").exists():
    add_sys_path(UTILITY_PATH)

from pykotor.common.misc import Game
from pykotor.resource.formats.ncs import NCS, NCSInstructionType, bytes_ncs, compile_nss, read_ncs


_COPY_INSTRUCTIONS = {
    NCSInstructionType.CPDOWNSP,
    NCSInstructionType.CPTOPSP,
    NCSInstructionType.CPDOWNBP,
    NCSInstructionType.CPTOPBP,
}


class TestNCSBinaryIO(unittest.TestCase):
    def roundtrip(self, ncs: NCS) -> tuple[bytearray, NCS]:
        data = bytes_ncs(ncs)
        loaded = read_ncs(data)
        self.assertEqual(data, bytes_ncs(loaded))
        return data, loaded

    def test_wide_copy_operands_roundtrip(self):
        ncs = NCS()
        ncs.add(NCSInstructionType.CPDOWNSP, [-24, 12])
        ncs.add(NCSInstructionType.CPTOPSP, [-16, 8])
        ncs.add(NCSInstructionType.CPDOWNBP, [-32, 12])
        ncs.add(NCSInstructionType.CPTOPBP, [-20, 8])

        data, loaded = self.roundtrip(ncs)

        self.assertEqual(len(data), int.from_bytes(data[9:13], byteorder="big"))
        self.assertEqual(
            [
                [-24, 12],
                [-16, 8],
                [-32, 12],
                [-20, 8],
            ],
            [instruction.args for instruction in loaded.instructions],
        )

    def test_signed_32_bit_operands_roundtrip(self):
        ncs = NCS()
        ncs.add(NCSInstructionType.CONSTI, [-1])
        ncs.add(NCSInstructionType.MOVSP, [-12])
        ncs.add(NCSInstructionType.INCISP, [-4])
        ncs.add(NCSInstructionType.DECISP, [-8])
        ncs.add(NCSInstructionType.INCIBP, [-16])
        ncs.add(NCSInstructionType.DECIBP, [-20])

        _, loaded = self.roundtrip(ncs)

        self.assertEqual(
            [[-1], [-12], [-4], [-8], [-16], [-20]],
            [instruction.args for instruction in loaded.instructions],
        )

    def test_object_constant_uses_full_32_bits(self):
        ncs = NCS()
        ncs.add(NCSInstructionType.CONSTO, [0xDEADBEEF])
        ncs.add(NCSInstructionType.RETN)

        data, loaded = self.roundtrip(ncs)

        self.assertEqual(13 + 6 + 2, len(data))
        self.assertEqual([0xDEADBEEF], loaded.instructions[0].args)
        self.assertEqual(NCSInstructionType.RETN, loaded.instructions[1].ins_type)

    def test_other_fixed_width_operands_roundtrip(self):
        ncs = NCS()
        ncs.add(NCSInstructionType.CONSTF, [1.25])
        ncs.add(NCSInstructionType.ACTION, [1234, 5])
        ncs.add(NCSInstructionType.DESTRUCT, [24, 8, 12])
        ncs.add(NCSInstructionType.STORE_STATE, [16, 104])
        ncs.add(NCSInstructionType.EQUALTT, [12])
        ncs.add(NCSInstructionType.NEQUALTT, [8])

        _, loaded = self.roundtrip(ncs)

        self.assertEqual([1.25], loaded.instructions[0].args)
        self.assertEqual([1234, 5], loaded.instructions[1].args)
        self.assertEqual([24, 8, 12], loaded.instructions[2].args)
        self.assertEqual([16, 104], loaded.instructions[3].args)
        self.assertEqual([12], loaded.instructions[4].args)
        self.assertEqual([8], loaded.instructions[5].args)

    def test_jump_offsets_roundtrip_in_both_directions(self):
        ncs = NCS()
        loop_start = ncs.add(NCSInstructionType.NOP)
        forward_jump = ncs.add(NCSInstructionType.JMP)
        ncs.add(NCSInstructionType.CONSTI, [123])
        loop_end = ncs.add(NCSInstructionType.NOP)
        backward_jump = ncs.add(NCSInstructionType.JMP, jump=loop_start)
        ncs.add(NCSInstructionType.RETN)
        forward_jump.jump = loop_end

        _, loaded = self.roundtrip(ncs)

        self.assertIs(loaded.instructions[3], loaded.instructions[1].jump)
        self.assertIs(loaded.instructions[0], loaded.instructions[4].jump)

    def test_compiled_vector_return_preserves_wide_copy_size(self):
        ncs = compile_nss(
            """
            vector MakeVector()
            {
                return [1.0, 2.0, 3.0];
            }

            void main()
            {
                vector value = MakeVector();
            }
            """,
            Game.K1,
        )

        _, loaded = self.roundtrip(ncs)
        wide_copies = [
            instruction.args
            for instruction in loaded.instructions
            if instruction.ins_type in _COPY_INSTRUCTIONS and instruction.args[1] == 12
        ]

        self.assertTrue(wide_copies)

    def test_compiled_struct_return_preserves_wide_copy_size(self):
        ncs = compile_nss(
            """
            struct Pair
            {
                int left;
                int right;
            };

            struct Pair MakePair()
            {
                struct Pair value;
                value.left = 1;
                value.right = 2;
                return value;
            }

            void main()
            {
                struct Pair value = MakePair();
            }
            """,
            Game.K1,
        )

        _, loaded = self.roundtrip(ncs)
        wide_copies = [
            instruction.args
            for instruction in loaded.instructions
            if instruction.ins_type in _COPY_INSTRUCTIONS and instruction.args[1] == 8
        ]

        self.assertTrue(wide_copies)


if __name__ == "__main__":
    unittest.main()
