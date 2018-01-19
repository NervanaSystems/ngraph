#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import pytest
import numpy as np

import pyngraph.util as util
from pyngraph import Type, Function
from pyngraph.runtime import Manager
from pyngraph.op import Acos, Asin, Cos, Sin
from pyngraph.op import Parameter, Maximum, Minimum, Reshape, Broadcast
from pyngraph.op import Add, Subtract, Multiply, Divide, Dot
from pyngraph.op import Constant, Abs, Exp, Log, Sum
from pyngraph.op import Greater, Less, Convert, Reduce
from pyngraph.op import OneHot, Negative


def make_backend_call_frame(function):

    manager = Manager.get(pytest.config.getoption('backend'));
    external = manager.compile(function)
    backend = manager.allocate_backend()
    cf = backend.make_call_frame(external)

    return backend, cf


def binary_op(op_str, a, b):

    if op_str == "+":
        return a + b
    elif op_str == "Add":
        return Add(a, b)
    elif op_str == "-":
        return a - b
    elif op_str == "Sub":
        return Subtract(a, b)
    elif op_str == "*":
        return a * b
    elif op_str == "Mul":
        return Multiply(a, b)
    elif op_str == "/":
        return a / b
    elif op_str == "Div":
        return Divide(a, b)
    elif op_str == "Dot":
        return Dot(a, b)
    elif op_str == "Maximum":
        return Maximum(a, b)
    elif op_str == "Minimum":
        return Minimum(a, b)


def binary_op_ref(op_str, a, b):

    if op_str == "+" or op_str == "Add":
        return a + b
    elif op_str == "-" or op_str == "Sub":
        return a - b
    elif op_str == "*" or op_str == "Mul":
        return a * b
    elif op_str == "/" or op_str == "Div":
        return a / b
    elif op_str == "Dot":
        return np.dot(a, b)
    elif op_str == "Maximum":
        return np.maximum(a, b)
    elif op_str == "Minimum":
        return np.minimum(a, b)


def binary_op_exec(op_str):

    element_type = Type.f32
    shape = [2,2]
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    parameter_list = [A, B]
    function = Function([binary_op(op_str, A, B)], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, shape)
    b = backend.make_primary_tensor_view(element_type, shape)
    result = backend.make_primary_tensor_view(element_type, shape)

    a.write(util.numpy_to_c(np.array([[1,6],[7,4]], dtype=np.float32)), 0, 16)
    b.write(util.numpy_to_c(np.array([[5,2],[3,8]], dtype=np.float32)), 0, 16)

    result_arr = np.array([[0, 0], [0, 0]], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 16)
    cf.call([a, b], [result])
    result.read(util.numpy_to_c(result_arr), 0, 16)

    a_arr = np.array([[1, 6], [7, 4]], dtype=np.float32)
    b_arr = np.array([[5, 2], [3, 8]], dtype=np.float32)
    result_arr_ref = binary_op_ref(op_str, a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_add():
    binary_op_exec("+")


def test_add_op():
    binary_op_exec("Add")


def test_sub():
    binary_op_exec("-")


def test_sub_op():
    binary_op_exec("Sub")


def test_mul():
    binary_op_exec("*")


def test_mul_op():
    binary_op_exec("Mul")


def test_div():
    binary_op_exec("/")


def test_div_op():
    binary_op_exec("Div")


def test_dot():
    binary_op_exec("Dot")


def test_maximum():
    binary_op_exec("Maximum")


def test_minimum():
    binary_op_exec("Minimum")


def test_add_with_mul():

    element_type = Type.f32
    shape = [2,2]
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    C = Parameter(element_type, shape)
    parameter_list = [A, B, C]
    function = Function([Multiply(Add(A,  B), C)], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, shape)
    b = backend.make_primary_tensor_view(element_type, shape)
    c = backend.make_primary_tensor_view(element_type, shape)
    result = backend.make_primary_tensor_view(element_type, shape)

    a.write(util.numpy_to_c(np.array([1,2,3,4], dtype=np.float32)), 0, 16)
    b.write(util.numpy_to_c(np.array([5,6,7,8], dtype=np.float32)), 0, 16)
    c.write(util.numpy_to_c(np.array([9,10,11,12], dtype=np.float32)), 0, 16)

    result_arr = np.array([0, 0, 0, 0], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 16)
    cf.call([a, b, c], [result])
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
    elif op_str == 'Cos':
        return Cos(a)
    elif op_str == 'log':
        return Log(a)
    elif op_str == 'exp':
        return Exp(a)
    elif op_str == 'negative':
        return Negative(a)
    elif op_str == 'Sin':
        return Sin(a)


def unary_op_ref(op_str, a):
    if op_str == 'Abs':
        return np.abs(a)
    elif op_str == 'Acos':
        return np.arccos(a)
    elif op_str == 'Asin':
        return np.arcsin(a)
    elif op_str == 'Cos':
        return np.cos(a)
    elif op_str == 'log':
        return np.log(a)
    elif op_str == 'exp':
        return np.exp(a)
    elif op_str == 'negative':
        return np.negative(a)
    elif op_str == 'Sin':
        return np.sin(a)


def unary_op_exec(op_str, input_list):

    if len(input_list) != 4:
        raise ValueError("Invalid list size: list length needs to be 4")
    element_type = Type.f32
    shape = [2,2]
    A = Parameter(element_type, shape)
    parameter_list = [A]
    function = Function([unary_op(op_str, A)], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, shape)
    result = backend.make_primary_tensor_view(element_type, shape)

    a.write(util.numpy_to_c(np.array(input_list, dtype=np.float32)), 0, 16)

    result_arr = np.array([0, 0, 0, 0], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 16)
    cf.call([a], [result])
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


def test_cos():
    input_list = [0, 0.7, 1.7, 3.4]
    op_str = 'Cos'
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


def test_sin():
    input_list = [0, 0.7, 1.7, 3.4]
    op_str = 'Sin'
    unary_op_exec(op_str, input_list)


def test_sum():

    element_type = Type.f32
    shape = [1, 4]
    A = Parameter(element_type, shape)
    parameter_list = [A]
    function = Function([Sum(A, {1})], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, shape)
    result = backend.make_primary_tensor_view(element_type, [1])

    a.write(util.numpy_to_c(np.array([1, 2, 3, 4], dtype=np.float32)), 0, 16)

    result_arr = np.array([0], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 4)
    cf.call([a], [result])
    result.read(util.numpy_to_c(result_arr), 0, 4)

    a_arr = np.array([1, 2, 3, 4], dtype=np.float32)
    result_arr_ref = np.sum(a_arr)

    assert np.allclose(result_arr[0], result_arr_ref)


def test_greater():

    element_type = Type.f32
    shape = [1,3]
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    parameter_list = [A, B]
    function = Function([Greater(A,  B)], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, shape)
    b = backend.make_primary_tensor_view(element_type, shape)
    result = backend.make_primary_tensor_view(Type.boolean, shape)

    a.write(util.numpy_to_c(np.array([1, 5, 3], dtype=np.float32)), 0, 12)
    b.write(util.numpy_to_c(np.array([2, 4, 6], dtype=np.float32)), 0, 12)

    result_arr = np.array([False, False, False], dtype=np.bool)
    result.write(util.numpy_to_c(result_arr), 0, 3)
    cf.call([a, b], [result])
    result.read(util.numpy_to_c(result_arr), 0, 3)

    a_arr = np.array([1, 5, 3], dtype=np.float32)
    b_arr = np.array([2, 4, 6], dtype=np.float32)
    result_arr_ref = np.greater(a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_less():

    element_type = Type.f32
    shape = [1,3]
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    parameter_list = [A, B]
    function = Function([Less(A,  B)], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, shape)
    b = backend.make_primary_tensor_view(element_type, shape)
    result = backend.make_primary_tensor_view(Type.boolean, shape)

    a.write(util.numpy_to_c(np.array([1, 5, 3], dtype=np.float32)), 0, 12)
    b.write(util.numpy_to_c(np.array([2, 4, 6], dtype=np.float32)), 0, 12)

    result_arr = np.array([False, False, False], dtype=np.bool)
    result.write(util.numpy_to_c(result_arr), 0, 3)
    cf.call([a, b], [result])
    result.read(util.numpy_to_c(result_arr), 0, 3)

    a_arr = np.array([1, 5, 3], dtype=np.float32)
    b_arr = np.array([2, 4, 6], dtype=np.float32)
    result_arr_ref = np.less(a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_reshape():

    element_type = Type.f32
    shape = [2,3]
    A = Parameter(element_type, shape)
    parameter_list = [A]
    function = Function([Reshape(A,  [0, 1], [3, 2])], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, shape)
    result = backend.make_primary_tensor_view(element_type, [3, 2])

    a.write(util.numpy_to_c(np.array([[1,2,3],[4,5,6]], dtype=np.float32)), 0, 24)

    result_arr = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 24)
    cf.call([a], [result])
    result.read(util.numpy_to_c(result_arr), 0, 24)

    a_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    result_arr_ref = np.reshape(a_arr, (3, 2))

    assert np.allclose(result_arr, result_arr_ref)


def test_convert():

    element_type = Type.f32
    shape = [1,3]
    A = Parameter(element_type, shape)
    parameter_list = [A]
    #f32 to boolean
    function = Function([Convert(A, Type.boolean)], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, shape)
    result = backend.make_primary_tensor_view(Type.boolean, shape)

    a.write(util.numpy_to_c(np.array([1, 5, 3], dtype=np.float32)), 0, 12)

    result_arr = np.array([False, False, False], dtype=np.bool)
    result.write(util.numpy_to_c(result_arr), 0, 3)
    cf.call([a], [result])
    result.read(util.numpy_to_c(result_arr), 0, 3)

    a_arr = np.array([1, 5, 3], dtype=np.float32)
    result_arr_ref = a_arr.astype(bool)
    assert np.allclose(result_arr, result_arr_ref)

    #f32 to i32
    function = Function([Convert(A, Type.i32)], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    result = backend.make_primary_tensor_view(Type.i32, shape)

    a.write(util.numpy_to_c(np.array([1.4, 5.5, 3.9], dtype=np.float32)), 0, 12)

    result_arr = np.array([0, 0, 0], dtype=np.int32)
    result.write(util.numpy_to_c(result_arr), 0, 12)
    cf.call([a], [result])
    result.read(util.numpy_to_c(result_arr), 0, 12)

    a_arr = np.array([1.4, 5.4, 3.9], dtype=np.float32)
    result_arr_ref = a_arr.astype(int)

    assert np.allclose(result_arr, result_arr_ref)


def test_broadcast():

    element_type = Type.f32
    A = Parameter(element_type, [3])
    parameter_list = [A]
    function = Function([Broadcast(A, [3, 3], {0})], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, [3])
    result = backend.make_primary_tensor_view(element_type, [3,3])

    a.write(util.numpy_to_c(np.array([1,2,3], dtype=np.float32)), 0, 12)

    result_arr = np.zeros((3,3), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 36)
    cf.call([a], [result])
    result.read(util.numpy_to_c(result_arr), 0, 36)

    a_arr = np.array([[0],[0],[0]], dtype=np.float32)
    b_arr = np.array([[1, 2, 3]], dtype=np.float32)
    result_arr_ref = np.add(a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_constant():

    element_type = Type.f32
    parameter_list = []
    function = Function([Constant(element_type, [3,3], list(range(9)))],
                                 parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    result = backend.make_primary_tensor_view(element_type, [3,3])

    result_arr = np.zeros((3,3), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 36)
    cf.call([], [result])
    result.read(util.numpy_to_c(result_arr), 0, 36)

    result_arr_ref = np.arange(9).reshape(3,3)

    assert np.allclose(result_arr, result_arr_ref)


def test_reduce():

    float_element_type = Type.f32

    AddParam1 = Parameter(float_element_type, [])
    AddParam2 = Parameter(float_element_type, [])
    constant_op = Constant(float_element_type, [], [0.])
    reduce_function = Function([Add(AddParam1, AddParam2)],
                                        [AddParam1, AddParam2], 'add')

    A = Parameter(float_element_type, [2, 2, 2])
    parameter_list = [A]

    function = Function([Reduce(A, constant_op, reduce_function, {0})],
                                 parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(float_element_type, [2, 2, 2])
    result = backend.make_primary_tensor_view(float_element_type, [2,2])

    a.write(util.numpy_to_c(np.arange(8, dtype=np.float32).reshape(2,2,2)), 0, 32)

    result_arr = np.zeros((2,2), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 16)
    cf.call([a], [result])
    result.read(util.numpy_to_c(result_arr), 0, 16)

    a_arr = np.arange(8).reshape(2,2,2)
    result_arr_ref = np.add.reduce(a_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_onehot():

    element_type = Type.f32
    A = Parameter(element_type, [3])
    parameter_list = [A]
    function = Function([OneHot(A, [3, 3], 0)], parameter_list, 'test')
    backend, cf = make_backend_call_frame(function)

    a = backend.make_primary_tensor_view(element_type, [3])
    result = backend.make_primary_tensor_view(element_type, [3,3])

    a.write(util.numpy_to_c(np.array([1,0,2], dtype=np.float32)), 0, 12)

    result_arr = np.zeros((3,3), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 0, 36)
    cf.call([a], [result])
    result.read(util.numpy_to_c(result_arr), 0, 36)

    a_arr = np.array([1,0,2])
    result_arr_ref = np.eye(3)[a_arr]

    assert np.allclose(result_arr, result_arr_ref)


if  __name__ == '__main__':
    test_cos()
