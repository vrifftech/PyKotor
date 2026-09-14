from __future__ import annotations

import struct

from copy import copy
from inspect import signature
from typing import TYPE_CHECKING, Any, NamedTuple

from pykotor.common.geometry import Vector3
from pykotor.common.script import DataType
from pykotor.common.scriptdefs import KOTOR_FUNCTIONS
from pykotor.resource.formats.ncs import NCSInstructionType
from pykotor.resource.formats.ncs.compiler.numeric import float32, int32
from pykotor.resource.formats.ncs.compiler.vm_arithmetic import (
    divide_vector_float32,
    k2_x86_shift_left,
    k2_x86_shift_right,
    k2_x86_unsigned_shift_right,
)
from pykotor.resource.formats.ncs.compiler.vm_values import (
    OBJECT_INVALID_ID,
    float32_bits_equal,
    ncs_strings_equal,
    resolve_object_literal,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pykotor.common.script import ScriptFunction
    from pykotor.resource.formats.ncs import NCS, NCSInstruction


class Interpreter:
    """Partial NCS interpreter with mocked engine calls.

    executing_object is a resolved unsigned handle; the default is OBJECT_INVALID_ID.
    """

    def __init__(
        self,
        ncs: NCS,
        *,
        functions: list[ScriptFunction] | None = None,
        executing_object: int = OBJECT_INVALID_ID,
    ):
        if type(executing_object) is not int:
            raise TypeError("executing_object must be an unsigned 32-bit integer handle")
        if not 0 <= executing_object <= 0xFFFFFFFF:
            raise ValueError("executing_object must be an unsigned 32-bit integer handle")
        self._executing_object = executing_object
        if not ncs.instructions:
            raise ValueError("Cannot execute an empty NCS program")
        self._ncs: NCS = ncs
        self._cursor: NCSInstruction | None = ncs.instructions[0]
        self._functions = KOTOR_FUNCTIONS if functions is None else functions
        self._instruction_indices = {id(ins): index for index, ins in enumerate(ncs.instructions)}
        self._stored_action: tuple[int, int, int] | None = None

        self._stack: Stack = Stack()
        self._returns: list[NCSInstruction | None] = [None]

        self._mocks: dict[str, Callable] = {}

        self.stack_snapshots: list[StackSnapshot] = []
        self.action_snapshots: list[ActionSnapshot] = []

    @property
    def executing_object(self) -> int:
        """The runtime handle used by OBJECT_SELF in this execution frame."""
        return self._executing_object

    def run(self, *, max_instructions: int = 100_000) -> None:
        """Execute instructions up to the configured budget."""
        if max_instructions <= 0:
            raise ValueError("Instruction budget must be positive")
        executed = 0
        while self._cursor is not None:
            if executed >= max_instructions:
                raise RuntimeError(f"NCS instruction budget exceeded ({max_instructions})")
            executed += 1
            index = self._instruction_indices[id(self._cursor)]
            jump_value = None


            if self._cursor.ins_type == NCSInstructionType.CONSTS:
                self._stack.add(DataType.STRING, self._cursor.args[0])

            elif self._cursor.ins_type == NCSInstructionType.CONSTI:
                self._stack.add(DataType.INT, self._cursor.args[0])

            elif self._cursor.ins_type == NCSInstructionType.CONSTF:
                self._stack.add(DataType.FLOAT, self._cursor.args[0])

            elif self._cursor.ins_type == NCSInstructionType.CONSTO:
                self._stack.add(
                    DataType.OBJECT,
                    resolve_object_literal(self._cursor.args[0], self.executing_object),
                )

            elif self._cursor.ins_type == NCSInstructionType.CPTOPSP:
                self._stack.copy_to_top(self._cursor.args[0], self._cursor.args[1])

            elif self._cursor.ins_type == NCSInstructionType.CPDOWNSP:
                self._stack.copy_down(self._cursor.args[0], self._cursor.args[1])

            elif self._cursor.ins_type == NCSInstructionType.ACTION:
                self.do_action(
                    self._functions[self._cursor.args[0]],
                    self._cursor.args[1],
                )

            elif self._cursor.ins_type == NCSInstructionType.MOVSP:
                self._stack.move(self._cursor.args[0])

            elif self._cursor.ins_type in {
                NCSInstructionType.ADDVV,
                NCSInstructionType.SUBVV,
                NCSInstructionType.MULVF,
                NCSInstructionType.MULFV,
                NCSInstructionType.DIVVF,
            }:
                self._stack.vector_op(self._cursor.ins_type)

            elif self._cursor.ins_type in {
                NCSInstructionType.ADDII,
                NCSInstructionType.ADDIF,
                NCSInstructionType.ADDFF,
                NCSInstructionType.ADDFI,
                NCSInstructionType.ADDSS,
            }:
                self._stack.addition_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.SUBII,
                NCSInstructionType.SUBIF,
                NCSInstructionType.SUBFF,
                NCSInstructionType.SUBFI,
            }:
                self._stack.subtraction_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.MULII,
                NCSInstructionType.MULIF,
                NCSInstructionType.MULFF,
                NCSInstructionType.MULFI,
            }:
                self._stack.multiplication_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.DIVII,
                NCSInstructionType.DIVIF,
                NCSInstructionType.DIVFF,
                NCSInstructionType.DIVFI,
            }:
                self._stack.division_op()

            elif self._cursor.ins_type == NCSInstructionType.MODII:
                self._stack.modulus_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.NEGI,
                NCSInstructionType.NEGF,
            }:
                self._stack.negation_op()

            elif self._cursor.ins_type == NCSInstructionType.COMPI:
                self._stack.bitwise_not_op()

            elif self._cursor.ins_type == NCSInstructionType.NOTI:
                self._stack.logical_not_op()

            elif self._cursor.ins_type == NCSInstructionType.LOGANDII:
                self._stack.logical_and_op()

            elif self._cursor.ins_type == NCSInstructionType.LOGORII:
                self._stack.logical_or_op()

            elif self._cursor.ins_type == NCSInstructionType.INCORII:
                self._stack.bitwise_or_op()

            elif self._cursor.ins_type == NCSInstructionType.EXCORII:
                self._stack.bitwise_xor_op()

            elif self._cursor.ins_type == NCSInstructionType.BOOLANDII:
                self._stack.bitwise_and_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.EQUALII,
                NCSInstructionType.EQUALFF,
                NCSInstructionType.EQUALSS,
                NCSInstructionType.EQUALOO,
                NCSInstructionType.EQUALEFFEFF,
                NCSInstructionType.EQUALEVTEVT,
                NCSInstructionType.EQUALLOCLOC,
                NCSInstructionType.EQUALTALTAL,
            }:
                self._stack.logical_equality_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.NEQUALII,
                NCSInstructionType.NEQUALFF,
                NCSInstructionType.NEQUALSS,
                NCSInstructionType.NEQUALOO,
                NCSInstructionType.NEQUALEFFEFF,
                NCSInstructionType.NEQUALEVTEVT,
                NCSInstructionType.NEQUALLOCLOC,
                NCSInstructionType.NEQUALTALTAL,
            }:
                self._stack.logical_inequality_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.GTII,
                NCSInstructionType.GTFF,
            }:
                self._stack.compare_greaterthan_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.GEQII,
                NCSInstructionType.GEQFF,
            }:
                self._stack.compare_greaterthanorequal_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.LTII,
                NCSInstructionType.LTFF,
            }:
                self._stack.compare_lessthan_op()

            elif self._cursor.ins_type in {
                NCSInstructionType.LEQII,
                NCSInstructionType.LEQFF,
            }:
                self._stack.compare_lessthanorequal_op()

            elif self._cursor.ins_type == NCSInstructionType.SHLEFTII:
                self._stack.bitwise_leftshift_op()

            elif self._cursor.ins_type == NCSInstructionType.SHRIGHTII:
                self._stack.bitwise_rightshift_op()

            elif self._cursor.ins_type == NCSInstructionType.USHRIGHTII:
                self._stack.bitwise_unsigned_rightshift_op()

            elif self._cursor.ins_type in {NCSInstructionType.EQUALTT, NCSInstructionType.NEQUALTT}:
                self._stack.structure_equality_op(
                    self._cursor.args[0], negate=self._cursor.ins_type == NCSInstructionType.NEQUALTT
                )

            elif self._cursor.ins_type == NCSInstructionType.DESTRUCT:
                self._stack.destruct(*self._cursor.args)

            elif self._cursor.ins_type == NCSInstructionType.INCIBP:
                self._stack.increment_bp(self._cursor.args[0])

            elif self._cursor.ins_type == NCSInstructionType.DECIBP:
                self._stack.decrement_bp(self._cursor.args[0])

            elif self._cursor.ins_type == NCSInstructionType.INCISP:
                self._stack.increment(self._cursor.args[0])

            elif self._cursor.ins_type == NCSInstructionType.DECISP:
                self._stack.decrement(self._cursor.args[0])

            elif self._cursor.ins_type == NCSInstructionType.RSADDI:
                self._stack.add(DataType.INT, 0)

            elif self._cursor.ins_type == NCSInstructionType.RSADDF:
                self._stack.add(DataType.FLOAT, 0)

            elif self._cursor.ins_type == NCSInstructionType.RSADDS:
                self._stack.add(DataType.STRING, "")

            elif self._cursor.ins_type == NCSInstructionType.RSADDO:
                self._stack.add(DataType.OBJECT, OBJECT_INVALID_ID)

            elif self._cursor.ins_type == NCSInstructionType.RSADDEFF:
                self._stack.add(DataType.EFFECT, 0)

            elif self._cursor.ins_type == NCSInstructionType.RSADDTAL:
                self._stack.add(DataType.TALENT, 0)

            elif self._cursor.ins_type == NCSInstructionType.RSADDLOC:
                self._stack.add(DataType.LOCATION, 0)

            elif self._cursor.ins_type == NCSInstructionType.RSADDEVT:
                self._stack.add(DataType.EVENT, 0)

            elif self._cursor.ins_type == NCSInstructionType.SAVEBP:
                self._stack.save_bp()

            elif self._cursor.ins_type == NCSInstructionType.RESTOREBP:
                self._stack.restore_bp()

            elif self._cursor.ins_type == NCSInstructionType.CPTOPBP:
                self._stack.copy_top_bp(self._cursor.args[0], self._cursor.args[1])

            elif self._cursor.ins_type == NCSInstructionType.CPDOWNBP:
                self._stack.copy_down_bp(self._cursor.args[0], self._cursor.args[1])

            elif self._cursor.ins_type == NCSInstructionType.JSR:
                index_return_to = index + 1
                return_to = self._ncs.instructions[index_return_to]
                self._returns.append(return_to)

            elif self._cursor.ins_type in {
                NCSInstructionType.JZ,
                NCSInstructionType.JNZ,
            }:
                jump_value = self._stack.pop()

            elif self._cursor.ins_type == NCSInstructionType.STORE_STATE:
                self.store_state()

            elif self._cursor.ins_type not in {
                NCSInstructionType.NOP, NCSInstructionType.JMP, NCSInstructionType.RETN
            }:
                raise NotImplementedError(f"Unsupported test-VM instruction: {self._cursor.ins_type.name}")

            self.stack_snapshots.append(
                StackSnapshot(self._cursor, self._stack.state()),
            )

            if self._cursor.ins_type == NCSInstructionType.RETN:
                return_to = self._returns.pop()
                self._cursor = return_to
                continue

            if (
                self._cursor.ins_type == NCSInstructionType.JMP
                or (self._cursor.ins_type == NCSInstructionType.JZ and jump_value == 0)
                or (self._cursor.ins_type == NCSInstructionType.JNZ and jump_value != 0)
                or self._cursor.ins_type == NCSInstructionType.JSR
            ):
                self._cursor = self._cursor.jump
            else:
                self._cursor = self._ncs.instructions[index + 1]

    def store_state(self):
        """Record capture metadata; ACTION copies the selected cells when consuming it."""
        index = self._instruction_indices[id(self._cursor)]
        global_bytes, local_bytes = self._cursor.args
        self._stored_action = (index, global_bytes, local_bytes)

    def _capture_action(self) -> ActionStackValue:
        if self._stored_action is None:
            raise ValueError("ACTION parameter has no preceding STORE_STATE")
        index, global_bytes, local_bytes = self._stored_action
        stack, base_pointer = self._stack.capture_state(global_bytes, local_bytes)
        after = self._ncs.instructions[index + 1].jump
        if after is None:
            raise ValueError("STORE_STATE must be followed by a jump over its action")
        end = self._instruction_indices[id(after)]
        # Preserve instruction identities so deferred jump targets remain valid.
        return ActionStackValue(
            self._ncs.instructions[index + 2 : end - 1],
            stack,
            self._ncs.instructions[index + 2],
            base_pointer,
            self.executing_object,
        )

    def run_stored_action(
        self,
        action: ActionStackValue,
        *,
        executing_object: int | None = None,
        max_instructions: int = 100_000,
    ) -> Interpreter:
        """Resume a captured action with its saved owner or an explicit override.

        Captured object values are unchanged; new OBJECT_SELF literals use that owner.
        """
        if action.entry is None:
            raise ValueError("Captured action has no entry instruction")
        owner = action.executing_object if executing_object is None else executing_object
        child = Interpreter(self._ncs, functions=self._functions, executing_object=owner)
        child._cursor = action.entry
        child._stack._stack = [copy(value) for value in action.stack]
        child._stack._bp = action.base_pointer
        child._mocks = self._mocks.copy()
        child.run(max_instructions=max_instructions)
        return child

    def do_action(self, function: ScriptFunction, args: int):
        args_snap = []
        for param in function.params[:args]:
            if param.datatype == DataType.ACTION:
                args_snap.append(StackObject(DataType.ACTION, self._capture_action()))
            elif param.datatype == DataType.VECTOR:
                # Components are pushed x, y, z and popped in reverse order.
                z = self._stack.pop().value
                y = self._stack.pop().value
                x = self._stack.pop().value
                args_snap.append(StackObject(DataType.VECTOR, Vector3(x, y, z)))
            else:
                args_snap.append(self._stack.pop())

        for param, arg in zip(function.params, args_snap):
            if param.datatype != arg.data_type:
                raise ValueError(
                    f"Invoked action '{function.name}' received the wrong data type "
                    f"for parameter '{param.name}' valued at '{arg}'."
                )

        value = None
        if function.name in self._mocks:
            value = self._mocks[function.name](*[arg.value for arg in args_snap])
        if function.returntype == DataType.VECTOR:
            self._stack.add(DataType.VECTOR, Vector3(0.0, 0.0, 0.0) if value is None else value)
        elif function.returntype != DataType.VOID:
            if function.returntype == DataType.OBJECT and value is None:
                # Mock results are runtime handles, not CONSTO encodings.
                value = OBJECT_INVALID_ID
            self._stack.add(function.returntype, value)
        self.action_snapshots.append(ActionSnapshot(function.name, args_snap, value))

    def print(self):
        for snap in self.stack_snapshots:
            print(snap.instruction, "\n", snap.stack, "\n")

    def set_mock(self, function_name: str, mock: Callable):
        function = next(
            (function for function in self._functions if function.name == function_name),
            None,
        )

        if function is None:
            msg = f"Function '{function_name}' does not exist."
            raise ValueError(msg)

        mock_param_count = len(signature(mock).parameters)
        routine_param_count = len(function.params)
        if mock_param_count != routine_param_count:
            msg = f"Function '{function_name}' expects {routine_param_count} parameters not {mock_param_count}."
            raise ValueError(msg)

        self._mocks[function_name] = mock

    def remove_mock(self, function_name: str):
        self._mocks.pop(function_name)


class StackV2:
    def __init__(self):
        self._stack: bytearray = bytearray()
        self._base_pointer: int = 0
        self._base_pointer_saved: list[int] = []
        self._stack_types: list = []

    def state(self) -> bytearray:
        return copy(self._stack)

    def copy_down(self, offset: int, size: int):
        stacksize = len(self._stack)
        copied = self._stack[stacksize - size : stacksize]
        self._stack[stacksize - offset : stacksize - offset + size] = copied

    def copy_to_top(self, offset: int, size: int):
        stacksize = len(self._stack)
        copied = self._stack[stacksize - offset : stacksize - offset + size]
        self._stack.extend(copied)

    def add(self, datatype: DataType, value: float | int):  # noqa: PYI041,RUF100
        if datatype not in {DataType.INT, DataType.FLOAT}:
            raise NotImplementedError
        if not isinstance(value, int):
            raise ValueError
        self._stack.extend(struct.pack("i", value))


class Stack:
    def __init__(self):
        self._stack: list[StackObject] = []
        self._bp: int = 0

    def state(self) -> list:
        return copy(self._stack)

    def add(self, data_type: DataType, value: Any):
        if data_type == DataType.VECTOR:
            for component in value:
                self.add(DataType.FLOAT, component)
            return
        if value is not None:
            if data_type == DataType.INT:
                value = int32(int(value))
            elif data_type == DataType.FLOAT:
                value = float32(value)
        self._stack.append(StackObject(data_type, value))

    def _stack_index(self, offset: int) -> int:
        if offset >= 0 or offset % 4 or -offset > self.stack_pointer():
            raise ValueError(f"Invalid SP-relative stack offset: {offset}")
        return offset // 4

    def _stack_index_bp(self, offset: int) -> int:
        index = (self._bp + offset) // 4
        if offset >= 0 or offset % 4 or index < 0 or index >= len(self._stack):
            raise ValueError(f"Invalid BP-relative stack offset: {offset}")
        return index

    def stack_pointer(self) -> int:
        return len(self._stack) * 4

    def base_pointer(self) -> int:
        return self._bp

    def peek(self, offset: int) -> Any:
        return self._stack[self._stack_index(offset)]

    def _range(self, start: int, size: int) -> slice:
        if size < 0 or size % 4 or start < 0 or start + size // 4 > len(self._stack):
            raise ValueError(f"Invalid stack range: cell {start}, {size} bytes")
        return slice(start, start + size // 4)

    def copy_to_top(self, offset: int, size: int):
        start = len(self._stack) + self._stack_index(offset)
        self._stack.extend(self._stack[self._range(start, size)])

    def copy_down(self, offset: int, size: int):
        start = len(self._stack) + self._stack_index(offset)
        target = self._range(start, size)
        source = self._range(len(self._stack) - size // 4, size)
        self._stack[target] = self._stack[source]

    def pop(self) -> Any:
        return self._stack.pop()

    def move(self, offset: int):
        if offset > 0 or offset % 4 or -offset > self.stack_pointer():
            raise ValueError(f"Invalid stack adjustment: {offset}")
        if offset:
            del self._stack[offset // 4 :]

    def copy_down_bp(self, offset: int, size: int):
        target = self._range(self._stack_index_bp(offset), size)
        source = self._range(len(self._stack) - size // 4, size)
        self._stack[target] = self._stack[source]

    def copy_top_bp(self, offset: int, size: int):
        self._stack.extend(self._stack[self._range(self._stack_index_bp(offset), size)])

    def save_bp(self):
        previous_bp = self._bp
        self._bp = self.stack_pointer()
        self.add(DataType.INT, previous_bp // 4)

    def restore_bp(self):
        previous_bp = self.pop()
        if previous_bp.data_type != DataType.INT:
            raise ValueError("RESTOREBP requires the saved integer BP cell on top")
        self._bp = previous_bp.value * 4

    def capture_state(self, global_bytes: int, local_bytes: int) -> tuple[list[StackObject], int]:
        if global_bytes == 0 and local_bytes == 0:
            # Zero sizes capture the entire state.
            global_bytes = self._bp
            local_bytes = self.stack_pointer() - self._bp
        if global_bytes < 0 or local_bytes < 0 or global_bytes % 4 or local_bytes % 4:
            raise ValueError("Captured stack sizes must be nonnegative multiples of four")
        if global_bytes > self._bp:
            raise ValueError("Captured global range extends below the base pointer")
        globals_range = self._range((self._bp - global_bytes) // 4, global_bytes)
        locals_range = self._range(len(self._stack) - local_bytes // 4, local_bytes)
        values = self._stack[globals_range] + self._stack[locals_range]
        return [copy(value) for value in values], global_bytes

    def destruct(self, total_size: int, keep_offset: int, keep_size: int):
        if keep_offset % 4 or keep_offset < 0 or keep_offset + keep_size > total_size:
            raise ValueError("Invalid DESTRUCT retained range")
        start = len(self._stack) - total_size // 4
        removed = self._range(start, total_size)
        kept = self._range(start + keep_offset // 4, keep_size)
        self._stack[removed] = self._stack[kept]

    @staticmethod
    def _cells_equal(left: StackObject, right: StackObject) -> bool:
        """Compare typed cells for scalar and struct equality."""
        if left.data_type != right.data_type:
            return False
        if left.value is None or right.value is None:
            return left.value is right.value
        if left.data_type == DataType.STRING:
            return ncs_strings_equal(left.value, right.value)
        if left.data_type == DataType.FLOAT:
            return float32_bits_equal(left.value, right.value)
        return left.value == right.value

    def structure_equality_op(self, size: int, *, negate: bool):
        right_start = len(self._stack) - size // 4
        left_start = right_start - size // 4
        left = self._stack[self._range(left_start, size)]
        right = self._stack[self._range(right_start, size)]
        equal = all(self._cells_equal(a, b) for a, b in zip(left, right))
        del self._stack[left_start:]
        self.add(DataType.INT, not equal if negate else equal)

    def _pop_vector(self) -> tuple[float, float, float]:
        z, y, x = self.pop(), self.pop(), self.pop()
        if any(component.data_type != DataType.FLOAT for component in (x, y, z)):
            raise ValueError("Vector operation requires three float cells")
        return x.value, y.value, z.value

    def vector_op(self, instruction: NCSInstructionType):
        if instruction in {NCSInstructionType.ADDVV, NCSInstructionType.SUBVV}:
            right = self._pop_vector()
            left = self._pop_vector()
            if instruction == NCSInstructionType.ADDVV:
                result = tuple(a + b for a, b in zip(left, right))
            else:
                result = tuple(a - b for a, b in zip(left, right))
        else:
            if instruction == NCSInstructionType.MULFV:
                vector = self._pop_vector()
                scalar = self.pop().value
            else:
                scalar = self.pop().value
                vector = self._pop_vector()
            if instruction == NCSInstructionType.DIVVF:
                result = divide_vector_float32(vector, scalar)
            else:
                result = tuple(component * scalar for component in vector)
        self.add(DataType.VECTOR, result)

    def increment(self, offset: int):
        index = self._stack_index(offset)
        self._stack[index] = copy(self._stack[index])
        self._stack[index].value = int32(self._stack[index].value + 1)

    def decrement(self, offset: int):
        index = self._stack_index(offset)
        self._stack[index] = copy(self._stack[index])
        self._stack[index].value = int32(self._stack[index].value - 1)

    def increment_bp(self, offset: int):
        index = self._stack_index_bp(offset)
        self._stack[index] = copy(self._stack[index])
        self._stack[index].value = int32(self._stack[index].value + 1)

    def decrement_bp(self, offset: int):
        index = self._stack_index_bp(offset)
        self._stack[index] = copy(self._stack[index])
        self._stack[index].value = int32(self._stack[index].value - 1)

    def _pop_arithmetic_operands(self) -> tuple[DataType, Any, Any]:
        """Pop right then left, converting mixed numeric operands to float32."""
        right = self.pop()
        left = self.pop()
        if DataType.FLOAT in {left.data_type, right.data_type}:
            return DataType.FLOAT, float32(left.value), float32(right.value)
        return left.data_type, left.value, right.value

    def addition_op(self):
        result_type, left, right = self._pop_arithmetic_operands()
        self.add(result_type, left + right)

    def subtraction_op(self):
        result_type, left, right = self._pop_arithmetic_operands()
        self.add(result_type, left - right)

    def multiplication_op(self):
        result_type, left, right = self._pop_arithmetic_operands()
        self.add(result_type, left * right)

    def division_op(self):
        result_type, left, right = self._pop_arithmetic_operands()
        if result_type == DataType.INT:
            self.add(DataType.INT, self._integer_quotient(left, right))
        else:
            self.add(DataType.FLOAT, left / right)

    def modulus_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        quotient = self._integer_quotient(value2.value, value1.value)
        self.add(DataType.INT, value2.value - quotient * value1.value)

    @staticmethod
    def _integer_quotient(dividend: int, divisor: int) -> int:
        # Integer division truncates toward zero.
        quotient = abs(dividend) // abs(divisor)
        return -quotient if (dividend < 0) != (divisor < 0) else quotient

    def negation_op(self):
        value1 = self._stack.pop()
        self.add(value1.data_type, -value1.value)

    def logical_not_op(self):
        value1 = self._stack.pop()
        self.add(DataType.INT, not value1.value)

    def logical_and_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(DataType.INT, bool(value2.value) and bool(value1.value))

    def logical_or_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(DataType.INT, bool(value2.value) or bool(value1.value))

    def logical_equality_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(DataType.INT, self._cells_equal(value2, value1))

    def logical_inequality_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(DataType.INT, not self._cells_equal(value2, value1))

    def bitwise_not_op(self):
        value1 = self._stack.pop()
        self.add(value1.data_type, ~value1.value)

    def bitwise_or_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(value1.data_type, value1.value | value2.value)

    def bitwise_xor_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(value1.data_type, value1.value ^ value2.value)

    def bitwise_and_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(value1.data_type, value1.value & value2.value)

    def bitwise_leftshift_op(self) -> None:
        count = self.pop().value
        value = self.pop().value
        self.add(DataType.INT, k2_x86_shift_left(value, count))

    def bitwise_rightshift_op(self) -> None:
        count = self.pop().value
        value = self.pop().value
        self.add(DataType.INT, k2_x86_shift_right(value, count))

    def bitwise_unsigned_rightshift_op(self) -> None:
        count = self.pop().value
        value = self.pop().value
        self.add(DataType.INT, k2_x86_unsigned_shift_right(value, count))

    def compare_greaterthan_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(DataType.INT, value2.value > value1.value)

    def compare_greaterthanorequal_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(DataType.INT, value2.value >= value1.value)

    def compare_lessthan_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(DataType.INT, value2.value < value1.value)

    def compare_lessthanorequal_op(self):
        value1 = self._stack.pop()
        value2 = self._stack.pop()
        self.add(DataType.INT, value2.value <= value1.value)


class StackObject:
    def __init__(self, data_type: DataType, value: Any):
        self.data_type: DataType = data_type
        self.value = value

    def __repr__(self):
        return f"{self.data_type.name}={self.value}"

    def __eq__(self, other: StackObject | object):
        if self is other:
            return True
        if isinstance(other, StackObject):
            return self.value == other.value
        return self.value == other


class ActionStackValue(NamedTuple):
    """Captured stack cells, continuation, and execution owner."""

    block: list[NCSInstruction]
    stack: list[StackObject]
    entry: NCSInstruction | None = None
    base_pointer: int = 0
    executing_object: int = OBJECT_INVALID_ID


class ActionSnapshot(NamedTuple):
    function_name: str
    arg_values: list
    return_value: Any


class StackSnapshot(NamedTuple):
    instruction: NCSInstruction
    stack: list[StackObject]


class EngineRoutineMock:
    def __init__(self, function: ScriptFunction, mock: Callable):
        self.function: ScriptFunction = function
        self.mock: Callable = mock
