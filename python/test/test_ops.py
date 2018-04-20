# ******************************************************************************
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
# flake8: noqa
from __future__ import absolute_import

import pytest
import numpy as np

from ngraph.impl import util
from ngraph.impl import Shape, Strides, CoordinateDiff, AxisSet, AxisVector, Coordinate
from ngraph.impl import Type, Function, NodeVector
from ngraph.impl.runtime import Backend
from ngraph.impl.op import Acos, Asin, Atan, Cos, Sin, Tan
from ngraph.impl.op import Cosh, Sinh, Tanh, Sqrt, Sign
from ngraph.impl.op import Power, Negative, Ceiling, Floor
from ngraph.impl.op import Parameter, Maximum, Minimum
from ngraph.impl.op import Add, Subtract, Multiply, Divide, Dot
from ngraph.impl.op import Constant, Abs, Exp, Log, Sum
from ngraph.impl.op import Greater, Less, Equal, NotEqual, GreaterEq, LessEq, Not
from ngraph.impl.op import OneHot, Broadcast, Reshape, Convert, Reduce
from ngraph.impl.op import Concat, Select
from ngraph.impl.op import Reverse, MaxPool, ReplaceSlice, Slice
from ngraph.impl.op import Convolution, ConvolutionBackpropData, ConvolutionBackpropFilters


def binary_op(op_str, a, b):

    if op_str == '+':
        return a + b
    elif op_str == 'Add':
        return Add(a, b)
    elif op_str == '-':
        return a - b
    elif op_str == 'Sub':
        return Subtract(a, b)
    elif op_str == '*':
        return a * b
    elif op_str == 'Mul':
        return Multiply(a, b)
    elif op_str == '/':
        return a / b
    elif op_str == 'Div':
        return Divide(a, b)
    elif op_str == 'Dot':
        return Dot(a, b)
    elif op_str == 'Equal':
        return Equal(a, b)
    elif op_str == 'Greater':
        return Greater(a, b)
    elif op_str == 'GreaterEq':
        return GreaterEq(a, b)
    elif op_str == 'Less':
        return Less(a, b)
    elif op_str == 'LessEq':
        return LessEq(a, b)
    elif op_str == 'Maximum':
        return Maximum(a, b)
    elif op_str == 'Minimum':
        return Minimum(a, b)
    elif op_str == 'NotEqual':
        return NotEqual(a, b)
    elif op_str == 'Power':
        return Power(a, b)


def binary_op_ref(op_str, a, b):

    if op_str == '+' or op_str == 'Add':
        return a + b
    elif op_str == '-' or op_str == 'Sub':
        return a - b
    elif op_str == '*' or op_str == 'Mul':
        return a * b
    elif op_str == '/' or op_str == 'Div':
        return a / b
    elif op_str == 'Dot':
        return np.dot(a, b)
    elif op_str == 'Equal':
        return np.equal(a, b)
    elif op_str == 'Greater':
        return np.greater(a, b)
    elif op_str == 'GreaterEq':
        return np.greater_equal(a, b)
    elif op_str == 'Less':
        return np.less(a, b)
    elif op_str == 'LessEq':
        return np.less_equal(a, b)
    elif op_str == 'Maximum':
        return np.maximum(a, b)
    elif op_str == 'Minimum':
        return np.minimum(a, b)
    elif op_str == 'NotEqual':
        return np.not_equal(a, b)
    elif op_str == 'Power':
        return np.power(a, b)


def binary_op_exec(op_str):

    element_type = Type.f32
    shape = Shape([2, 2])
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    parameter_list = [A, B]
    function = Function(NodeVector([binary_op(op_str, A, B)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    b = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, shape)

    a.write(util.numpy_to_c(np.array([[1, 6], [7, 4]], dtype=np.float32)), 0, 16)
    b.write(util.numpy_to_c(np.array([[5, 2], [3, 8]], dtype=np.float32)), 0, 16)

    result_arr = np.array([[0, 0], [0, 0]], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 16)
    backend.call(function, [result], [a, b])
    result.read(util.numpy_to_c(result_arr), 0, 16)

    a_arr = np.array([[1, 6], [7, 4]], dtype=np.float32)
    b_arr = np.array([[5, 2], [3, 8]], dtype=np.float32)
    result_arr_ref = binary_op_ref(op_str, a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def binary_op_comparison(op_str):

    element_type = Type.f32
    shape = Shape([2, 2])
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    parameter_list = [A, B]
    function = Function(NodeVector([binary_op(op_str, A, B)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    b = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(Type.boolean, shape)

    a.write(util.numpy_to_c(np.array([[1, 5], [3, 2]], dtype=np.float32)), 0, 16)
    b.write(util.numpy_to_c(np.array([[2, 4], [3, 1]], dtype=np.float32)), 0, 16)

    result_arr = np.array([[False, False], [False, False]], dtype=np.bool)
    result.write(util.numpy_to_c(result_arr), 0, 4)
    backend.call(function, [result], [a, b])
    result.read(util.numpy_to_c(result_arr), 0, 4)

    a_arr = np.array([[1, 5], [3, 2]], dtype=np.float32)
    b_arr = np.array([[2, 4], [3, 1]], dtype=np.float32)
    result_arr_ref = binary_op_ref(op_str, a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_add():
    binary_op_exec('+')


def test_add_op():
    binary_op_exec('Add')


def test_sub():
    binary_op_exec('-')


def test_sub_op():
    binary_op_exec('Sub')


def test_mul():
    binary_op_exec('*')


def test_mul_op():
    binary_op_exec('Mul')


def test_div():
    binary_op_exec('/')


def test_div_op():
    binary_op_exec('Div')


def test_dot():
    binary_op_exec('Dot')


def test_maximum():
    binary_op_exec('Maximum')


def test_minimum():
    binary_op_exec('Minimum')


def test_power():
    binary_op_exec('Power')


def test_greater():
    binary_op_comparison('Greater')


def test_greater_eq():
    binary_op_comparison('GreaterEq')


def test_less():
    binary_op_comparison('Less')


def test_less_eq():
    binary_op_comparison('LessEq')


def test_not_equal():
    binary_op_comparison('NotEqual')


def test_add_with_mul():

    element_type = Type.f32
    shape = Shape([2, 2])
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    C = Parameter(element_type, shape)
    parameter_list = [A, B, C]
    function = Function(NodeVector([Multiply(Add(A, B), C)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    b = backend.create_tensor(element_type, shape)
    c = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, shape)

    a.write(util.numpy_to_c(np.array([1, 2, 3, 4], dtype=np.float32)), 0, 16)
    b.write(util.numpy_to_c(np.array([5, 6, 7, 8], dtype=np.float32)), 0, 16)
    c.write(util.numpy_to_c(np.array([9, 10, 11, 12], dtype=np.float32)), 0, 16)

    result_arr = np.array([0, 0, 0, 0], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 16)
    backend.call(function, [result], [a, b, c])
    result.read(util.numpy_to_c(result_arr), 0, 16)

    a_arr = np.array([1, 2, 3, 4], dtype=np.float32)
    b_arr = np.array([5, 6, 7, 8], dtype=np.float32)
    c_arr = np.array([9, 10, 11, 12], dtype=np.float32)
    result_arr_ref = (a_arr + b_arr) * c_arr

    assert np.allclose(result_arr, result_arr_ref)


def unary_op(op_str, a):
    if op_str == 'Abs':
        return Abs(a)
    elif op_str == 'Acos':
        return Acos(a)
    elif op_str == 'Asin':
        return Asin(a)
    elif op_str == 'Atan':
        return Atan(a)
    elif op_str == 'Ceiling':
        return Ceiling(a)
    elif op_str == 'Cos':
        return Cos(a)
    elif op_str == 'Cosh':
        return Cosh(a)
    elif op_str == 'Floor':
        return Floor(a)
    elif op_str == 'log':
        return Log(a)
    elif op_str == 'exp':
        return Exp(a)
    elif op_str == 'negative':
        return Negative(a)
    elif op_str == 'Reverse':
        return Reverse(a, AxisSet({1}))
    elif op_str == 'Sign':
        return Sign(a)
    elif op_str == 'Sin':
        return Sin(a)
    elif op_str == 'Sinh':
        return Sinh(a)
    elif op_str == 'Sqrt':
        return Sqrt(a)
    elif op_str == 'Tan':
        return Tan(a)
    elif op_str == 'Tanh':
        return Tanh(a)


def unary_op_ref(op_str, a):
    if op_str == 'Abs':
        return np.abs(a)
    elif op_str == 'Acos':
        return np.arccos(a)
    elif op_str == 'Asin':
        return np.arcsin(a)
    elif op_str == 'Atan':
        return np.arctan(a)
    elif op_str == 'Ceiling':
        return np.ceil(a)
    elif op_str == 'Cos':
        return np.cos(a)
    elif op_str == 'Cosh':
        return np.cosh(a)
    elif op_str == 'Floor':
        return np.floor(a)
    elif op_str == 'log':
        return np.log(a)
    elif op_str == 'exp':
        return np.exp(a)
    elif op_str == 'negative':
        return np.negative(a)
    elif op_str == 'Reverse':
        return np.fliplr(a)
    elif op_str == 'Sign':
        return np.sign(a)
    elif op_str == 'Sin':
        return np.sin(a)
    elif op_str == 'Sinh':
        return np.sinh(a)
    elif op_str == 'Sqrt':
        return np.sqrt(a)
    elif op_str == 'Tan':
        return np.tan(a)
    elif op_str == 'Tanh':
        return np.tanh(a)


def unary_op_exec(op_str, input_list):
    """
    input_list needs to have deep length of 4
    """
    element_type = Type.f32
    shape = Shape(np.array(input_list).shape)
    shape_np = np.array(input_list).shape
    A = Parameter(element_type, shape)
    parameter_list = [A]
    function = Function(NodeVector([unary_op(op_str, A)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, shape)

    a.write(util.numpy_to_c(np.array(input_list, dtype=np.float32)), 0, 16)

    result_arr = np.zeros(shape_np, dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 16)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 16)

    a_arr = np.array(input_list, dtype=np.float32)
    result_arr_ref = unary_op_ref(op_str, a_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_abs():
    input_list = [-1, 0, 1, 2]
    op_str = 'Abs'
    unary_op_exec(op_str, input_list)


def test_acos():
    input_list = [-1, 0, 0.5, 1]
    op_str = 'Acos'
    unary_op_exec(op_str, input_list)


def test_asin():
    input_list = [-1, 0, 0.5, 1]
    op_str = 'Asin'
    unary_op_exec(op_str, input_list)


def test_atan():
    input_list = [-1, 0, 0.5, 1]
    op_str = 'Atan'
    unary_op_exec(op_str, input_list)


def test_ceiling():
    input_list = [0.5, 0, 0.4, 0.5]
    op_str = 'Ceiling'
    unary_op_exec(op_str, input_list)


def test_cos():
    input_list = [0, 0.7, 1.7, 3.4]
    op_str = 'Cos'
    unary_op_exec(op_str, input_list)


def test_cosh():
    input_list = [-1, 0., 0.5, 1]
    op_str = 'Cosh'
    unary_op_exec(op_str, input_list)


def test_floor():
    input_list = [-0.5, 0, 0.4, 0.5]
    op_str = 'Floor'
    unary_op_exec(op_str, input_list)


def test_log():
    input_list = [1, 2, 3, 4]
    op_str = 'log'
    unary_op_exec(op_str, input_list)


def test_exp():
    input_list = [-1, 0, 1, 2]
    op_str = 'exp'
    unary_op_exec(op_str, input_list)


def test_negative():
    input_list = [-1, 0, 1, 2]
    op_str = 'negative'
    unary_op_exec(op_str, input_list)


def test_sign():
    input_list = [-1, 0, 0.5, 1]
    op_str = 'Sign'
    unary_op_exec(op_str, input_list)


def test_sin():
    input_list = [0, 0.7, 1.7, 3.4]
    op_str = 'Sin'
    unary_op_exec(op_str, input_list)


def test_sinh():
    input_list = [-1, 0., 0.5, 1]
    op_str = 'Sinh'
    unary_op_exec(op_str, input_list)


def test_sqrt():
    input_list = [0., 0.5, 1, 2]
    op_str = 'Sqrt'
    unary_op_exec(op_str, input_list)


def test_tan():
    input_list = [-np.pi / 4, 0, np.pi / 8, np.pi / 8]
    op_str = 'Tan'
    unary_op_exec(op_str, input_list)


def test_tanh():
    input_list = [-1, 0, 0.5, 1]
    op_str = 'Tanh'
    unary_op_exec(op_str, input_list)


@pytest.config.gpu_skip(reason="Not implemented")
def test_reverse():
    input_list = [[-1, 0], [0.5, 1]]
    op_str = 'Reverse'
    unary_op_exec(op_str, input_list)


def test_not():
    element_type = Type.boolean
    shape = Shape([2])
    A = Parameter(element_type, shape)
    parameter_list = [A]
    function = Function(NodeVector([Not(A)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(Type.boolean, shape)

    a.write(util.numpy_to_c(np.array([True, False], dtype=np.bool)), 0, 2)

    result_arr = np.array([False, False], dtype=np.bool)
    result.write(util.numpy_to_c(result_arr), 0, 2)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 2)

    a_arr = np.array([True, False], dtype=np.bool)
    result_arr_ref = np.logical_not(a_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_sum():

    element_type = Type.f32
    shape = Shape([1, 4])
    A = Parameter(element_type, shape)
    parameter_list = [A]
    function = Function(NodeVector([Sum(A, AxisSet({1}))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, Shape([1]))

    a.write(util.numpy_to_c(np.array([1, 2, 3, 4], dtype=np.float32)), 0, 16)

    result_arr = np.array([0], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 4)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 4)

    a_arr = np.array([1, 2, 3, 4], dtype=np.float32)
    result_arr_ref = np.sum(a_arr)

    assert np.allclose(result_arr[0], result_arr_ref)


def test_reshape():

    element_type = Type.f32
    shape = Shape([2, 3])
    A = Parameter(element_type, shape)
    parameter_list = [A]
    function = Function(NodeVector([Reshape(A, AxisVector([0, 1]), Shape([3, 2]))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, Shape([3, 2]))

    a.write(util.numpy_to_c(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)), 0, 24)

    result_arr = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 24)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 24)

    a_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    result_arr_ref = np.reshape(a_arr, (3, 2))

    assert np.allclose(result_arr, result_arr_ref)


def test_convert():

    element_type = Type.f32
    shape = Shape([1, 3])
    A = Parameter(element_type, shape)
    parameter_list = [A]
    # f32 to boolean
    function = Function(NodeVector([Convert(A, Type.boolean)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(Type.boolean, shape)

    a.write(util.numpy_to_c(np.array([1, 5, 3], dtype=np.float32)), 0, 12)

    result_arr = np.array([False, False, False], dtype=np.bool)
    result.write(util.numpy_to_c(result_arr), 0, 3)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 3)

    a_arr = np.array([1, 5, 3], dtype=np.float32)
    result_arr_ref = a_arr.astype(bool)
    assert np.allclose(result_arr, result_arr_ref)

    # f32 to i32
    function = Function(NodeVector([Convert(A, Type.i32)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    result = backend.create_tensor(Type.i32, shape)

    a.write(util.numpy_to_c(np.array([1.4, 5.5, 3.9], dtype=np.float32)), 0, 12)

    result_arr = np.array([0, 0, 0], dtype=np.int32)
    result.write(util.numpy_to_c(result_arr), 0, 12)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 12)

    a_arr = np.array([1.4, 5.4, 3.9], dtype=np.float32)
    result_arr_ref = a_arr.astype(int)

    assert np.allclose(result_arr, result_arr_ref)


def test_broadcast():

    element_type = Type.f32
    A = Parameter(element_type, Shape([3]))
    parameter_list = [A]
    function = Function(NodeVector([Broadcast(A, Shape([3, 3]), AxisSet({0}))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, Shape([3]))
    result = backend.create_tensor(element_type, Shape([3, 3]))

    a.write(util.numpy_to_c(np.array([1, 2, 3], dtype=np.float32)), 0, 12)

    result_arr = np.zeros((3, 3), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 36)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 36)

    a_arr = np.array([[0], [0], [0]], dtype=np.float32)
    b_arr = np.array([[1, 2, 3]], dtype=np.float32)
    result_arr_ref = np.add(a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_constant():

    element_type = Type.f32
    parameter_list = []
    function = Function(NodeVector([Constant(element_type, Shape([3, 3]), list(range(9)))]),
                        parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    result = backend.create_tensor(element_type, Shape([3, 3]))

    result_arr = np.zeros((3, 3), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 36)
    backend.call(function, [result], [])
    result.read(util.numpy_to_c(result_arr), 0, 36)

    result_arr_ref = np.arange(9).reshape(3, 3)

    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_reduce():

    float_element_type = Type.f32

    AddParam1 = Parameter(float_element_type, Shape([]))
    AddParam2 = Parameter(float_element_type, Shape([]))
    constant_op = Constant(float_element_type, Shape([]), [0.])
    reduce_function = Function(NodeVector([Add(AddParam1, AddParam2)]),
                               [AddParam1, AddParam2], 'add')

    A = Parameter(float_element_type, Shape([2, 2, 2]))
    parameter_list = [A]

    function = Function(NodeVector([Reduce(A, constant_op, reduce_function, AxisSet({0}))]),
                        parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(float_element_type, Shape([2, 2, 2]))
    result = backend.create_tensor(float_element_type, Shape([2, 2]))

    a.write(util.numpy_to_c(np.arange(8, dtype=np.float32).reshape(2, 2, 2)), 0, 32)

    result_arr = np.zeros((2, 2), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 16)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 16)

    a_arr = np.arange(8).reshape(2, 2, 2)
    result_arr_ref = np.add.reduce(a_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_onehot():

    element_type = Type.f32
    A = Parameter(element_type, Shape([3]))
    parameter_list = [A]
    function = Function(NodeVector([OneHot(A, Shape([3, 3]), 0)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, Shape([3]))
    result = backend.create_tensor(element_type, Shape([3, 3]))

    a.write(util.numpy_to_c(np.array([1, 0, 2], dtype=np.float32)), 0, 12)

    result_arr = np.zeros((3, 3), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 36)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 36)

    a_arr = np.array([1, 0, 2])
    result_arr_ref = np.eye(3)[a_arr]

    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_concat():

    element_type = Type.f32
    A = Parameter(element_type, Shape([1, 2]))
    B = Parameter(element_type, Shape([1, 2]))
    C = Parameter(element_type, Shape([1, 2]))
    parameter_list = [A, B, C]
    axis = 0
    function = Function(NodeVector([Concat(NodeVector([A, B, C]), axis)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, Shape([1, 2]))
    b = backend.create_tensor(element_type, Shape([1, 2]))
    c = backend.create_tensor(element_type, Shape([1, 2]))
    result = backend.create_tensor(element_type, Shape([3, 2]))

    a.write(util.numpy_to_c(np.array([1, 2], dtype=np.float32)), 0, 8)
    b.write(util.numpy_to_c(np.array([5, 6], dtype=np.float32)), 0, 8)
    c.write(util.numpy_to_c(np.array([7, 8], dtype=np.float32)), 0, 8)

    result_arr = np.zeros(6, dtype=np.float32).reshape(3, 2)
    result.write(util.numpy_to_c(result_arr), 0, 24)
    backend.call(function, [result], [a, b, c])
    result.read(util.numpy_to_c(result_arr), 0, 24)

    a_arr = np.array([[1, 2]], dtype=np.float32)
    b_arr = np.array([[5, 6]], dtype=np.float32)
    c_arr = np.array([[7, 8]], dtype=np.float32)
    result_arr_ref = np.concatenate((a_arr, b_arr, c_arr), axis)

    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_select():

    element_type = Type.f32
    A = Parameter(Type.boolean, Shape([1, 2]))
    B = Parameter(element_type, Shape([1, 2]))
    C = Parameter(element_type, Shape([1, 2]))
    parameter_list = [A, B, C]

    function = Function(NodeVector([Select(A, B, C)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(Type.boolean, Shape([1, 2]))
    b = backend.create_tensor(element_type, Shape([1, 2]))
    c = backend.create_tensor(element_type, Shape([1, 2]))
    result = backend.create_tensor(element_type, Shape([1, 2]))

    a.write(util.numpy_to_c(np.array([[True, False]], dtype=np.bool)), 0, 2)
    b.write(util.numpy_to_c(np.array([[5, 6]], dtype=np.float32)), 0, 8)
    c.write(util.numpy_to_c(np.array([[7, 8]], dtype=np.float32)), 0, 8)

    result_arr = np.array([[0, 0]], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 8)
    backend.call(function, [result], [a, b, c])
    result.read(util.numpy_to_c(result_arr), 0, 8)

    result_arr_ref = np.array([[5, 8]])

    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_slice():

    element_type = Type.f32
    shape = Shape([6, 6])
    A = Parameter(element_type, shape)
    parameter_list = [A]

    input_arr = np.arange(36, dtype=np.float32).reshape(6, 6)
    lower_bounds = [1, 1]
    upper_bounds = [5, 5]

    function = Function(NodeVector([Slice(A, Coordinate(lower_bounds),
                                   Coordinate(upper_bounds))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, Shape([4, 4]))

    a.write(util.numpy_to_c(input_arr), 0, 36*4)

    result_arr = np.zeros(16, dtype=np.float32).reshape(4, 4)
    result.write(util.numpy_to_c(result_arr), 0, 16*4)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 64)

    result_arr_ref = input_arr[lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]

    assert np.allclose(result_arr, result_arr_ref)


    #test with strides
    strides = [1, 2]

    function = Function(NodeVector([Slice(A, Coordinate(lower_bounds), Coordinate(upper_bounds),
                        Strides(strides))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    result = backend.create_tensor(element_type, Shape([4, 2]))
    result_arr = np.zeros(8, dtype=np.float32).reshape(4, 2)

    result.write(util.numpy_to_c(result_arr), 0, 8*4)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 32)

    result_arr_ref = result_arr_ref[::strides[0], ::strides[1]]

    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_replace_slice():

    element_type = Type.f32
    A = Parameter(element_type, Shape([6, 4]))
    B = Parameter(element_type, Shape([3, 2]))
    parameter_list = [A, B]

    input_arr_a = np.zeros(24, dtype=np.float32).reshape(6, 4)
    input_arr_b = np.ones(6, dtype=np.float32).reshape(3, 2)
    lower_bounds = [0, 1]
    upper_bounds = [3, 3]

    function = Function(NodeVector([ReplaceSlice(A, B, Coordinate(lower_bounds),
                        Coordinate(upper_bounds))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, Shape([6, 4]))
    b = backend.create_tensor(element_type, Shape([3, 2]))
    result = backend.create_tensor(element_type, Shape([6, 4]))

    a.write(util.numpy_to_c(input_arr_a), 0, 24*4)
    b.write(util.numpy_to_c(input_arr_b), 0, 6*4)

    result_arr = np.zeros(24, dtype=np.float32).reshape(6, 4)
    result.write(util.numpy_to_c(result_arr), 0, 24*4)
    backend.call(function, [result], [a, b])
    result.read(util.numpy_to_c(result_arr), 0, 24*4)

    result_arr_ref = np.copy(input_arr_a)
    result_arr_ref[lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]] = input_arr_b

    assert np.allclose(result_arr, result_arr_ref)

    #test with strides
    lower_bounds = [0, 0]
    upper_bounds = [5, 3]
    strides = [2, 2]

    function = Function(NodeVector([ReplaceSlice(A, B, Coordinate(lower_bounds),
                        Coordinate(upper_bounds), Strides(strides))]),
                        parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    backend.call(function, [result], [a, b])
    result.read(util.numpy_to_c(result_arr), 0, 24*4)

    result_arr_ref = np.copy(input_arr_a)
    result_arr_ref[::strides[0], ::strides[1]] = input_arr_b

    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_max_pool():

    #test 1d
    element_type = Type.f32
    shape = Shape([1, 1, 10])
    A = Parameter(element_type, shape)
    parameter_list = [A]

    input_arr = np.arange(10, dtype=np.float32).reshape(1, 1, 10)
    window_shape = [3]

    function = Function(NodeVector([MaxPool(A, Shape(window_shape))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, Shape([1, 1, 8]))

    a.write(util.numpy_to_c(input_arr), 0, 10*4)

    result_arr = np.zeros(8, dtype=np.float32).reshape(1, 1, 8)
    result.write(util.numpy_to_c(result_arr), 0, 8*4)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 32)

    result_arr_ref = (np.arange(8) + 2).reshape(1, 1, 8)
    assert np.allclose(result_arr, result_arr_ref)

    #test 1d with strides
    strides = [2]

    function = Function(NodeVector([MaxPool(A, Shape(window_shape), Strides(strides))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    size = 4
    result = backend.create_tensor(element_type, Shape([1, 1, size]))
    result_arr = np.zeros(size, dtype=np.float32).reshape(1, 1, size)

    result.write(util.numpy_to_c(result_arr), 0, size*4)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, size*4)

    result_arr_ref = ((np.arange(size) + 1) * 2).reshape(1, 1, size)
    assert np.allclose(result_arr, result_arr_ref)

    #test 2d
    element_type = Type.f32
    shape = Shape([1, 1, 10, 10])
    A = Parameter(element_type, shape)
    parameter_list = [A]

    input_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    window_shape = [3, 3]

    function = Function(NodeVector([MaxPool(A, Shape(window_shape))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, Shape([1, 1, 8, 8]))

    a.write(util.numpy_to_c(input_arr), 0, 10*10*4)

    result_arr = np.zeros(64, dtype=np.float32).reshape(1, 1, 8, 8)
    result.write(util.numpy_to_c(result_arr), 0, 8*8*4)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, 8*8*4)

    result_arr_ref = ((np.arange(100).reshape(10, 10))[2:, 2:]).reshape(1, 1, 8, 8)
    assert np.allclose(result_arr, result_arr_ref)

    #test 2d with strides
    strides = [2, 2]

    function = Function(NodeVector([MaxPool(A, Shape(window_shape), Strides(strides))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    size = 4
    result = backend.create_tensor(element_type, Shape([1, 1, size, size]))
    result_arr = np.zeros(size*size, dtype=np.float32).reshape(1, 1, size, size)

    result.write(util.numpy_to_c(result_arr), 0, size*size*4)
    backend.call(function, [result], [a])
    result.read(util.numpy_to_c(result_arr), 0, size*size*4)

    result_arr_ref = ((np.arange(100).reshape(10, 10))[2::2, 2::2]).reshape(1, 1, size, size)
    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def convolution2d(image, filterit, strides=(1, 1), dilation=(1, 1), padding_below=(0, 0),
                  padding_above=(0, 0), data_dilation=(1, 1)):

    def dilate(arr, dil=(1, 1)):
        m, n = arr.shape
        new_m, new_n = (m - 1) * dil[0] + 1, (n - 1) * dil[1] + 1
        new_arr = np.zeros(new_m * new_n, dtype=np.float32).reshape(new_m, new_n)
        for i in range(m):
            for j in range(n):
                new_arr[dil[0] * i][dil[1] * j] = arr[i][j]
        return new_arr

    i_m, i_n = image.shape
    new_image = np.zeros((i_m + padding_below[0] + padding_above[0]) * \
                         (i_n + padding_below[1] + padding_above[1]),
                         dtype=np.float32).reshape(i_m + padding_below[0] + padding_above[0],
                                                   i_n + padding_below[1] + padding_above[1])
    new_image[padding_below[0] : padding_below[0] + i_m,
              padding_below[1] : padding_below[1] + i_n] = image
    image = new_image
    image = image if data_dilation[0] == data_dilation[1] == 1 else dilate(image, data_dilation)
    i_m, i_n = image.shape

    filterit = filterit if dilation[0] == dilation[1] == 1 else dilate(filterit, dilation)
    f_m, f_n = filterit.shape

    #result_shape
    r_m = i_m - f_m + 1
    r_n = i_n - f_n + 1
    r_m //= strides[0]
    r_n //= strides[1]

    result = np.zeros(r_m * r_n, dtype=np.float32).reshape(r_m, r_n)

    for i in range(r_m):
        for j in range(r_n):
            sub_m = image[i * strides[0] : i * strides[0] + f_m,
                          j * strides[1] : j * strides[1] + f_n]
            result[i][j] = np.sum(sub_m * filterit)
    return result


@pytest.config.gpu_skip(reason="Not implemented")
def test_convolution():

    element_type = Type.f32
    image_shape = Shape([1, 1, 16, 16])
    filter_shape = Shape([1, 1, 3, 3])
    A = Parameter(element_type, image_shape)
    B = Parameter(element_type, filter_shape)
    parameter_list = [A, B]

    image_arr = np.arange(-128, 128, 1, dtype=np.float32).reshape(1, 1, 16, 16)
    filter_arr = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
    filter_arr[0][0][0][0] = -1
    filter_arr[0][0][1][1] = -1
    filter_arr[0][0][2][2] = -1
    filter_arr[0][0][0][2] = -1
    filter_arr[0][0][2][0] = -1
    result_arr = np.zeros(196, dtype=np.float32).reshape(1, 1, 14, 14)

    function = Function(NodeVector([Convolution(A, B)]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 0, 16*16*4)
    b.write(util.numpy_to_c(filter_arr), 0, 3*3*4)

    result = backend.create_tensor(element_type, Shape([1, 1, 14, 14]))
    result.write(util.numpy_to_c(result_arr), 0, 14*14*4)
    backend.call(function, [result], [a, b])
    result.read(util.numpy_to_c(result_arr), 0, 14*14*4)

    result_arr_ref = convolution2d(image_arr[0][0], filter_arr[0][0]).reshape(1, 1, 14, 14)
    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_convolution_with_strides():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    A = Parameter(element_type, image_shape)
    B = Parameter(element_type, filter_shape)
    parameter_list = [A, B]

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = np.zeros(9, dtype=np.float32).reshape(1, 1, 3, 3)
    filter_arr[0][0][1][1] = 1
    strides = [2, 2]

    function = Function(NodeVector([Convolution(A, B, Strides(strides))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 0, 10*10*4)
    b.write(util.numpy_to_c(filter_arr), 0, 3*3*4)

    result_arr = np.zeros(16, dtype=np.float32).reshape(1, 1, 4, 4)
    result = backend.create_tensor(element_type, Shape([1, 1, 4, 4]))
    result.write(util.numpy_to_c(result_arr), 0, 4*4*4)
    backend.call(function, [result], [a, b])

    result.read(util.numpy_to_c(result_arr), 0, 4*4*4)
    result_arr_ref = convolution2d(image_arr[0][0], filter_arr[0][0], strides).reshape(1, 1, 4, 4)
    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_convolution_with_filter_dilation():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    A = Parameter(element_type, image_shape)
    B = Parameter(element_type, filter_shape)
    parameter_list = [A, B]

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
    strides = [1, 1]
    dilation = [2, 2]

    function = Function(NodeVector([Convolution(A, B, Strides(strides), Strides(dilation))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 0, 10*10*4)
    b.write(util.numpy_to_c(filter_arr), 0, 3*3*4)

    result_arr = np.zeros(36, dtype=np.float32).reshape(1, 1, 6, 6)
    result = backend.create_tensor(element_type, Shape([1, 1, 6, 6]))
    result.write(util.numpy_to_c(result_arr), 0, 6*6*4)
    backend.call(function, [result], [a, b])

    result.read(util.numpy_to_c(result_arr), 0, 6*6*4)
    result_arr_ref = convolution2d(image_arr[0][0], filter_arr[0][0], strides,
                                   dilation).reshape(1, 1, 6, 6)
    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_convolution_with_padding():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    A = Parameter(element_type, image_shape)
    B = Parameter(element_type, filter_shape)
    parameter_list = [A, B]

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = np.zeros(9, dtype=np.float32).reshape(1, 1, 3, 3)
    filter_arr[0][0][1][1] = 1
    strides = [1, 1]
    dilation = [2, 2]
    padding_below = [0, 0]
    padding_above = [0, 0]

    function = Function(NodeVector([Convolution(A, B, Strides(strides), Strides(dilation),
                        CoordinateDiff(padding_below), CoordinateDiff(padding_above))]),
                        parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 0, 10*10*4)
    b.write(util.numpy_to_c(filter_arr), 0, 3*3*4)

    result_arr = np.zeros(36, dtype=np.float32).reshape(1, 1, 6, 6)
    result = backend.create_tensor(element_type, Shape([1, 1, 6, 6]))
    result.write(util.numpy_to_c(result_arr), 0, 6*6*4)
    backend.call(function, [result], [a, b])

    result.read(util.numpy_to_c(result_arr), 0, 6*6*4)
    result_arr_ref = convolution2d(image_arr[0][0], filter_arr[0][0], strides,
                                   dilation, padding_below,
                                   padding_above).reshape(1, 1, 6, 6)
    assert np.allclose(result_arr, result_arr_ref)

    # test with non-zero padding
    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    A = Parameter(element_type, image_shape)
    B = Parameter(element_type, filter_shape)
    parameter_list = [A, B]

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = (np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)) * -1
    filter_arr[0][0][1][1] = 1
    strides = [1, 1]
    dilation = [2, 2]
    padding_below = [2, 1]
    padding_above = [1, 2]

    function = Function(NodeVector([Convolution(A, B, Strides(strides), Strides(dilation),
                        CoordinateDiff(padding_below), CoordinateDiff(padding_above))]),
                        parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 0, 10*10*4)
    b.write(util.numpy_to_c(filter_arr), 0, 3*3*4)

    result_arr = np.zeros(81, dtype=np.float32).reshape(1, 1, 9, 9)
    result = backend.create_tensor(element_type, Shape([1, 1, 9, 9]))
    result.write(util.numpy_to_c(result_arr), 0, 9*9*4)
    backend.call(function, [result], [a, b])

    result.read(util.numpy_to_c(result_arr), 0, 9*9*4)
    result_arr_ref = convolution2d(image_arr[0][0], filter_arr[0][0], strides,
                                   dilation, padding_below,
                                   padding_above).reshape(1, 1, 9, 9)
    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_convolution_with_data_dilation():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    A = Parameter(element_type, image_shape)
    B = Parameter(element_type, filter_shape)
    parameter_list = [A, B]

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
    strides = [1, 1]
    dilation = [1, 1]
    padding_below = [0, 0]
    padding_above = [0, 0]
    data_dilation = [2, 2]

    function = Function(NodeVector([Convolution(A, B, Strides(strides), Strides(dilation),
                                    CoordinateDiff(padding_below), CoordinateDiff(padding_above),
                                    Strides(data_dilation))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 0, 10*10*4)
    b.write(util.numpy_to_c(filter_arr), 0, 3*3*4)

    result_arr = np.zeros(17*17, dtype=np.float32).reshape(1, 1, 17, 17)
    result = backend.create_tensor(element_type, Shape([1, 1, 17, 17]))
    result.write(util.numpy_to_c(result_arr), 0, 17*17*4)
    backend.call(function, [result], [a, b])

    result.read(util.numpy_to_c(result_arr), 0, 17*17*4)
    result_arr_ref = convolution2d(image_arr[0][0], filter_arr[0][0], strides,
                                   dilation, padding_below, padding_above,
                                   data_dilation).reshape(1, 1, 17, 17)
    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_convolutionBackpropData():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    output_shape = Shape([1, 1, 17, 17])

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
    window_strides = [1, 1]
    window_dilation = [1, 1]
    padding_below = [0, 0]
    padding_above = [0, 0]
    data_dilation = [2, 2]

    output_arr = convolution2d(image_arr[0][0], filter_arr[0][0], window_strides,
                               window_dilation, padding_below, padding_above,
                               data_dilation).reshape(1, 1, 17, 17)

    A = Parameter(element_type, filter_shape)
    B = Parameter(element_type, output_shape)
    parameter_list = [A, B]

    function = Function(NodeVector([ConvolutionBackpropData(image_shape, A, B, Strides(window_strides), Strides(window_dilation),
                                     CoordinateDiff(padding_below), CoordinateDiff(padding_above),
                                     Strides(data_dilation))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, filter_shape)
    b = backend.create_tensor(element_type, output_shape)

    a.write(util.numpy_to_c(filter_arr), 0, 3*3*4)
    b.write(util.numpy_to_c(output_arr), 0, 17*17*4)

    result_arr = np.zeros(10*10, dtype=np.float32).reshape(1, 1, 10, 10)
    result = backend.create_tensor(element_type, Shape([1, 1, 10, 10]))
    result.write(util.numpy_to_c(result_arr), 0, 10*10*4)
    backend.call(function, [result], [a, b])

    result.read(util.numpy_to_c(result_arr), 0, 10*10*4)
    result_arr_ref = np.array(
        [[[[  22,   60,   70,   80,   90,  100,  110,  120,  130,   54.],
           [ 105,  275,  300,  325,  350,  375,  400,  425,  450,  185.],
           [ 205,  525,  550,  575,  600,  625,  650,  675,  700,  285.],
           [ 305,  775,  800,  825,  850,  875,  900,  925,  950,  385.],
           [ 405, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200,  485.],
           [ 505, 1275, 1300, 1325, 1350, 1375, 1400, 1425, 1450,  585.],
           [ 605, 1525, 1550, 1575, 1600, 1625, 1650, 1675, 1700,  685.],
           [ 705, 1775, 1800, 1825, 1850, 1875, 1900, 1925, 1950,  785.],
           [ 805, 2025, 2050, 2075, 2100, 2125, 2150, 2175, 2200,  885.],
           [ 342,  860,  870,  880,  890,  900,  910,  920,  930,  374.]]]])
    assert np.allclose(result_arr, result_arr_ref)


@pytest.config.gpu_skip(reason="Not implemented")
def test_convolutionBackpropFilters():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    output_shape = Shape([1, 1, 17, 17])

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
    window_strides = [1, 1]
    window_dilation = [1, 1]
    padding_below = [0, 0]
    padding_above = [0, 0]
    data_dilation = [2, 2]

    output_arr = convolution2d(image_arr[0][0], filter_arr[0][0], window_strides,
                               window_dilation, padding_below, padding_above,
                               data_dilation).reshape(1, 1, 17, 17)

    A = Parameter(element_type, image_shape)
    B = Parameter(element_type, output_shape)
    parameter_list = [A, B]

    function = Function(NodeVector([ConvolutionBackpropFilters(A, filter_shape, B, Strides(window_strides), Strides(window_dilation),
                                     CoordinateDiff(padding_below),CoordinateDiff(padding_above),
                                     Strides(data_dilation))]), parameter_list, 'test')
    backend = Backend.create(pytest.config.getoption('backend'))

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, output_shape)

    a.write(util.numpy_to_c(image_arr), 0, 10*10*4)
    b.write(util.numpy_to_c(output_arr), 0, 17*17*4)

    result_arr = np.zeros(3*3, dtype=np.float32).reshape(1, 1, 3, 3)
    result = backend.create_tensor(element_type, Shape([1, 1, 3, 3]))
    result.write(util.numpy_to_c(result_arr), 0, 3*3*4)
    backend.call(function, [result], [a, b])

    result.read(util.numpy_to_c(result_arr), 0, 3*3*4)
    result_arr_ref = np.array(
        [[[[ 923832,  413952,  939870.],
           [ 425832,  190752,  432960.],
           [1084212,  485232, 1100250.]]]])
    assert np.allclose(result_arr, result_arr_ref)
