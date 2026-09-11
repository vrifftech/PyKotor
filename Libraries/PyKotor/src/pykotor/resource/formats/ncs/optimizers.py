from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from pykotor.resource.formats.ncs.ncs_data import NCSInstructionType, NCSOptimizer

if TYPE_CHECKING:
    from pykotor.resource.formats.ncs.ncs_data import NCS, NCSInstruction


class RemoveNopOptimizer(NCSOptimizer):
    """NCS Compiler uses NOP instructions as stubs to simplify the compilation process however as their name suggests
    they do not perform any actual function. This optimizer removes all occurrences of NOP instructions from the
    compiled script.
    """  # noqa: D205

    def optimize(self, ncs: NCS):
        """Optimizes a neural circuit specification by removing NOP instructions.

        Args:
        ----
            ncs: NCS - The neural circuit specification to optimize

        Processing Logic:
        ----------------
            - Finds all NOP instructions in the NCS
            - For each NOP, finds all links jumping to it and updates them to jump to the next instruction instead
            - Removes all NOP instructions from the NCS instruction list.
        """
        nops: list[NCSInstruction] = [inst for inst in ncs.instructions if inst.ins_type == NCSInstructionType.NOP]

        # Process instructions which jump to a NOP and set them to jump to the proceeding instruction instead
        for nop in nops:
            nop_index: int = ncs.instructions.index(nop)
            for link in ncs.links_to(nop):
                link.jump = ncs.instructions[nop_index + 1]

        # It is now safe to remove all NOP instructions
        ncs.instructions = [inst for inst in ncs.instructions if inst.ins_type != NCSInstructionType.NOP]


class RemoveMoveSPEqualsZeroOptimizer(NCSOptimizer):
    def __init__(self):
        super().__init__()

    def optimize(self, ncs: NCS):
        """Optimizes an NCS script by removing unnecessary MOVSP=0 instructions.

        Args:
        ----
            ncs (NCS): The NCS script to optimize

        Processing Logic:
        ----------------
            - Finds all MOVSP=0 instructions
            - Changes any jumps to those instructions to jump to the next instruction instead
            - Removes all MOVSP=0 instructions from the program.
        """
        movsp0: list[NCSInstruction] = [inst for inst in ncs.instructions if inst.ins_type == NCSInstructionType.MOVSP and inst.args[0] == 0]

        # Process instructions which jump to a MOVSP=0 and set them to jump to the proceeding instruction instead
        for op in movsp0:
            nop_index: int = ncs.instructions.index(op)
            for link in ncs.links_to(op):
                link.jump = ncs.instructions[nop_index + 1]

        # It is now safe to remove all MOVSP=0 instructions
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
    """Remove instructions that cannot be reached from the script loader.

    For compiled NSS this is equivalent to BioWare's safe dead-function
    elimination: user functions are emitted before final reachability is known,
    then functions that cannot be reached from the loader/call graph disappear.
    The graph walk also removes any other truly unreachable instruction ranges.
    """

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
                # A subroutine call reaches both the callee and the instruction
                # following the call once the callee returns.
                add_jump_target()
                checking.append(index + 1)
            elif instruction.ins_type == NCSInstructionType.JMP:
                add_jump_target()
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
