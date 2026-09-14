from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from pykotor.resource.formats.ncs.ncs_data import NCSInstructionType, NCSOptimizer

if TYPE_CHECKING:
    from pykotor.resource.formats.ncs.ncs_data import NCS, NCSInstruction


class RemoveNopOptimizer(NCSOptimizer):
    """Remove symbolic NOP labels and redirect their incoming jumps."""  # noqa: D205

    def optimize(self, ncs: NCS):
        """Remove NOP instructions while preserving jump targets."""
        nops: list[NCSInstruction] = [inst for inst in ncs.instructions if inst.ins_type == NCSInstructionType.NOP]

        for nop in nops:
            nop_index: int = ncs.instructions.index(nop)
            for link in ncs.links_to(nop):
                link.jump = ncs.instructions[nop_index + 1]

        ncs.instructions = [inst for inst in ncs.instructions if inst.ins_type != NCSInstructionType.NOP]


class RemoveMoveSPEqualsZeroOptimizer(NCSOptimizer):
    def __init__(self):
        super().__init__()

    def optimize(self, ncs: NCS):
        """Remove zero-sized stack moves while preserving jump targets."""
        movsp0: list[NCSInstruction] = [inst for inst in ncs.instructions if inst.ins_type == NCSInstructionType.MOVSP and inst.args[0] == 0]

        for op in movsp0:
            nop_index: int = ncs.instructions.index(op)
            for link in ncs.links_to(op):
                link.jump = ncs.instructions[nop_index + 1]

        for inst in ncs.instructions.copy():
            if inst.ins_type == NCSInstructionType.MOVSP and inst.args[0] == 0:
                ncs.instructions.remove(inst)
                self.instructions_cleared += 1


class MergeAdjacentMoveSPOptimizer(NCSOptimizer):
    def optimize(self, ncs: NCS) -> NoReturn:
        raise NotImplementedError


class RemoveJMPToAdjacentOptimizer(NCSOptimizer):
    def optimize(self, ncs: NCS) -> NoReturn:
        raise NotImplementedError


class RemoveUnusedBlocksOptimizer(NCSOptimizer):
    """Remove instructions unreachable from entry, calls, and deferred continuations."""

    def optimize(self, ncs: NCS):
        instructions = ncs.instructions
        if not instructions:
            return

        index_by_instruction = {instruction: index for index, instruction in enumerate(instructions)}
        reachable_indices: set[int] = set()
        checking: list[int] = [0]

        while checking:
            index = checking.pop()
            if index < 0 or index >= len(instructions) or index in reachable_indices:
                continue

            reachable_indices.add(index)
            instruction = instructions[index]

            def add_jump_target() -> None:
                if instruction.jump is not None:
                    target = index_by_instruction.get(instruction.jump)
                    if target is not None:
                        checking.append(target)

            if instruction.ins_type in {NCSInstructionType.JZ, NCSInstructionType.JNZ}:
                add_jump_target()
                checking.append(index + 1)
            elif instruction.ins_type == NCSInstructionType.JSR:
                add_jump_target()
                checking.append(index + 1)
            elif instruction.ins_type == NCSInstructionType.JMP:
                add_jump_target()
            elif instruction.ins_type == NCSInstructionType.STORE_STATE:
                # The deferred entry follows the jump that skips its body.
                checking.append(index + 1)
                checking.append(index + 2)
            elif instruction.ins_type == NCSInstructionType.RETN:
                continue
            else:
                checking.append(index + 1)

        original_count = len(instructions)
        ncs.instructions = [
            instruction
            for index, instruction in enumerate(instructions)
            if index in reachable_indices
        ]
        self.instructions_cleared += original_count - len(ncs.instructions)


class RemoveUnusedGlobalsInStackOptimizer(NCSOptimizer):
    def optimize(self, ncs: NCS) -> NoReturn:
        raise NotImplementedError
