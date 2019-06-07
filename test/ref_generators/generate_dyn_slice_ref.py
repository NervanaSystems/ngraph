#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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

#
# Test case generator for DynSlice op.
#
# TODO(amprocte): try to get compile time of the generated code down, refactor
# to use parameterized gtests.
#

import sys
import numpy as np

def make_iterable(x):
    try:
        _ = iter(x)
    except TypeError as _:
        return [x]
    return x

def print_lb_values(slices):
    slices = make_iterable(slices)

    strs = []

    for sl in slices:
        try:
            x = int(sl)
            strs.append(str(x))
        except TypeError as _:
            if isinstance(sl, slice) and sl.start is not None:
                strs.append(str(sl.start))
            else:
                strs.append('0')
    return ','.join(strs)

def print_ub_values(slices):
    slices = make_iterable(slices)

    strs = []

    for sl in slices:
        if isinstance(sl, slice) and sl.stop is not None:
            strs.append(str(sl.stop))
        else:
            strs.append('0')
    return ','.join(strs)

def print_stride_values(slices):
    slices = make_iterable(slices)

    strs = []

    for sl in slices:
        if isinstance(sl, slice) and sl.step is not None:
            strs.append(str(sl.step))
        else:
            strs.append('1')
    return ','.join(strs)

def print_lb_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        if isinstance(sl, slice) and sl.start is None:
            mask_strs.append(str(i))
        i += 1
    return ','.join(mask_strs)

def print_ub_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        if isinstance(sl, slice) and sl.stop is None:
            mask_strs.append(str(i))
        i += 1
    return ','.join(mask_strs)

def print_new_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        if sl is None:
            mask_strs.append(str(i))
        i += 1
    return ','.join(mask_strs)

def print_shrink_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        try:
            _ = int(sl)
            mask_strs.append(str(i))
        except TypeError as _:
            pass
        i += 1
    return ','.join(mask_strs)

def print_ellipsis_mask_axes(slices):
    slices = make_iterable(slices)

    mask_strs = []
    i = 0

    for sl in slices:
        if sl is Ellipsis:
            mask_strs.append(str(i))
        i += 1
    return ','.join(mask_strs)

def np_dt_to_c(dtype):
    if dtype=='int8':
        return 'int8_t'
    elif dtype=='uint8':
        return 'uint8_t'
    elif dtype=='int16':
        return 'int16_t'
    elif dtype=='uint16':
        return 'uint16_t'
    elif dtype=='int32':
        return 'int32_t'
    elif dtype=='uint32':
        return 'uint32_t'
    elif dtype=='int64':
        return 'int64_t'
    elif dtype=='uint64':
        return 'uint64_t'
    elif dtype=='float16':
        return 'float16'
    elif dtype=='float32':
        return 'float'
    elif dtype=='float64':
        return 'double'
    elif dtype=='bool':
        return 'char'
    else:
        raise ValueError('Unsupported numpy data type: %s' % dtype)

def np_dt_to_ng(dtype):
    if dtype=='int8':
        return 'element::i8'
    elif dtype=='uint8':
        return 'element::u8'
    elif dtype=='int16':
        return 'element::i16'
    elif dtype=='uint16':
        return 'element::u16'
    elif dtype=='int32':
        return 'element::i32'
    elif dtype=='uint32':
        return 'element::u32'
    elif dtype=='int64':
        return 'element::i64'
    elif dtype=='uint64':
        return 'element::u64'
    elif dtype=='float16':
        return 'element::f16'
    elif dtype=='float32':
        return 'element::f32'
    elif dtype=='float64':
        return 'element::f64'
    elif dtype=='bool':
        return 'element::boolean'
    else:
        raise ValueError('Unsupported numpy data type: %s' % dtype)

def print_values(values):
    values = make_iterable(values)
    strs = []

    for v in values:
        strs.append(str(v))

    return ','.join(strs)

def print_shape(dims):
    dims = make_iterable(dims)
    strs = []

    for d in dims:
        strs.append(str(d))

    return 'Shape{' + ','.join(strs) + '}'

#
# Class to intercept indexing operations and write an nGraph C++ test case. The
# generated test case will ensure that the output is identical to that which
# would be produced by numpy on a "linspaced" array of the given shape and
# dtype. If numpy throws an exception when the slice is attempted, the test
# checks that nGraph throws some exception somewhere in graph construction or
# execution, but does not attempt to make sure that the exception is the
# "correct" one.
#
# Example usage:
#
#    w = SliceTestWriter(stream=sys.stdout)
#
#    # behave as if slicing a 4x5x6 input array of data type int32
#    w.set_shape(4,5,6)
#    w.set_dtype('int32')
#
#    # generate test cases for various behaviors, writing C++ code to sys.stdout
#    w[0,:,:]
#    w[...,-1:-3,:]
#    w[-3,...,-1:-6:-3]
#    w[1,2,3,4]        # note: in numpy this throws exception (too many
#                      # indices), so generated test case will check that
#                      # nGraph throws some exception too.
#
class SliceTestWriter:
    def __init__(self, shape=(), dtype='int32', stream=sys.stdout):
        self._shape = shape
        n_elems = np.prod(shape)
        self._dtype = dtype
        self._stream = stream

    def __getitem__(self, slices):
        self.write_test(slices)

    def write_test(self, slices):
        data_in = np.linspace(0,np.prod(self._shape)-1,np.prod(self._shape),dtype=self._dtype).reshape(self._shape)

        n_slices = 1
        try:
            n_slices = len(slices)
        except TypeError:
            pass

        try:
            data_out = data_in.__getitem__(slices)
        except Exception as e:
            self._stream.write('    {\n')
            self._stream.write('        auto arg = std::make_shared<op::Parameter>(%s, %s);\n' % (np_dt_to_ng(self._dtype), print_shape(self._shape)))
            self._stream.write('        auto lb = std::make_shared<op::Parameter>(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('        auto ub = std::make_shared<op::Parameter>(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('        auto strides = std::make_shared<op::Parameter>(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('\n')
            self._stream.write('        std::vector<%s> input_values(%s);\n' % (np_dt_to_c(self._dtype), np.prod(self._shape)))
            self._stream.write('        std::iota(input_values.begin(), input_values.end(), static_cast<%s>(0));\n' % np_dt_to_c(self._dtype))
            self._stream.write('        std::vector<int64_t> lb_values{%s};\n' % print_lb_values(slices))
            self._stream.write('        std::vector<int64_t> ub_values{%s};\n' % print_ub_values(slices))
            self._stream.write('        std::vector<int64_t> strides_values{%s};\n' % print_stride_values(slices))
            self._stream.write('        AxisSet lb_mask{%s};\n' % print_lb_mask_axes(slices))
            self._stream.write('        AxisSet ub_mask{%s};\n' % print_ub_mask_axes(slices))
            self._stream.write('        AxisSet new_mask{%s};\n' % print_new_mask_axes(slices))
            self._stream.write('        AxisSet shrink_mask{%s};\n' % print_shrink_mask_axes(slices))
            self._stream.write('        AxisSet ellipsis_mask{%s};\n' % print_ellipsis_mask_axes(slices))
            self._stream.write('\n')
            self._stream.write('        // numpy threw: %s\n' % type(e))
            self._stream.write('        EXPECT_ANY_THROW({\n');
            self._stream.write('            auto slice = std::make_shared<op::DynSlice>(arg, lb, ub, strides, lb_mask, ub_mask, new_mask, shrink_mask, ellipsis_mask);\n')
            self._stream.write('\n')
            self._stream.write('            auto f = std::make_shared<Function>(NodeVector{slice}, ParameterVector{arg, lb, ub, strides});\n')
            self._stream.write('\n')
            self._stream.write('            auto backend = runtime::Backend::create("${BACKEND_NAME}",true);\n')
            self._stream.write('            auto ex = backend->compile(f);\n')
            self._stream.write('\n')
            self._stream.write('            auto input_arg = backend->create_tensor(%s, %s);\n' % (np_dt_to_ng(self._dtype), print_shape(self._shape)))
            self._stream.write('            auto input_lb = backend->create_tensor(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('            auto input_ub = backend->create_tensor(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('            auto input_strides = backend->create_tensor(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('            copy_data(input_arg, input_values);\n')
            self._stream.write('            copy_data(input_lb, lb_values);\n')
            self._stream.write('            copy_data(input_ub, ub_values);\n')
            self._stream.write('            copy_data(input_strides, strides_values);\n')
            self._stream.write('\n')
            self._stream.write('            auto output = backend->create_dynamic_tensor(%s, PartialShape::dynamic());\n' % np_dt_to_ng(self._dtype))
            self._stream.write('\n')
            self._stream.write('            ex->call({output}, {input_arg, input_lb, input_ub, input_strides});\n')
            self._stream.write('        });\n')
            self._stream.write('    }\n')
        else:
            self._stream.write('    {\n')
            self._stream.write('        auto arg = std::make_shared<op::Parameter>(%s, %s);\n' % (np_dt_to_ng(self._dtype), print_shape(self._shape)))
            self._stream.write('        auto lb = std::make_shared<op::Parameter>(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('        auto ub = std::make_shared<op::Parameter>(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('        auto strides = std::make_shared<op::Parameter>(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('\n')
            self._stream.write('        std::vector<%s> input_values(%s);\n' % (np_dt_to_c(self._dtype), np.prod(self._shape)))
            self._stream.write('        std::iota(input_values.begin(), input_values.end(), static_cast<%s>(0));\n' % np_dt_to_c(self._dtype))
            self._stream.write('        std::vector<int64_t> lb_values{%s};\n' % print_lb_values(slices))
            self._stream.write('        std::vector<int64_t> ub_values{%s};\n' % print_ub_values(slices))
            self._stream.write('        std::vector<int64_t> strides_values{%s};\n' % print_stride_values(slices))
            self._stream.write('        AxisSet lb_mask{%s};\n' % print_lb_mask_axes(slices))
            self._stream.write('        AxisSet ub_mask{%s};\n' % print_ub_mask_axes(slices))
            self._stream.write('        AxisSet new_mask{%s};\n' % print_new_mask_axes(slices))
            self._stream.write('        AxisSet shrink_mask{%s};\n' % print_shrink_mask_axes(slices))
            self._stream.write('        AxisSet ellipsis_mask{%s};\n' % print_ellipsis_mask_axes(slices))
            self._stream.write('\n')
            self._stream.write('        auto slice = std::make_shared<op::DynSlice>(arg, lb, ub, strides, lb_mask, ub_mask, new_mask, shrink_mask, ellipsis_mask);\n')
            self._stream.write('\n')
            self._stream.write('        auto f = std::make_shared<Function>(NodeVector{slice}, ParameterVector{arg, lb, ub, strides});\n')
            self._stream.write('\n')
            self._stream.write('        auto backend = runtime::Backend::create("${BACKEND_NAME}",true);\n')
            self._stream.write('        auto ex = backend->compile(f);\n')
            self._stream.write('\n')
            self._stream.write('        auto input_arg = backend->create_tensor(%s, %s);\n' % (np_dt_to_ng(self._dtype), print_shape(self._shape)))
            self._stream.write('        auto input_lb = backend->create_tensor(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('        auto input_ub = backend->create_tensor(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('        auto input_strides = backend->create_tensor(element::i64, %s);\n' % print_shape((n_slices,)))
            self._stream.write('        copy_data(input_arg, input_values);\n')
            self._stream.write('        copy_data(input_lb, lb_values);\n')
            self._stream.write('        copy_data(input_ub, ub_values);\n')
            self._stream.write('        copy_data(input_strides, strides_values);\n')
            self._stream.write('\n')
            self._stream.write('        auto output = backend->create_dynamic_tensor(%s, PartialShape::dynamic());\n' % np_dt_to_ng(self._dtype))
            self._stream.write('\n')
            self._stream.write('        ex->call({output}, {input_arg, input_lb, input_ub, input_strides});\n')
            self._stream.write('\n')
            self._stream.write('        EXPECT_EQ(output->get_element_type(), (%s));\n' % np_dt_to_ng(self._dtype))
            self._stream.write('        EXPECT_EQ(output->get_shape(), (%s));\n' % print_shape(data_out.shape))
            self._stream.write('\n')
            self._stream.write('        auto output_values = read_vector<%s>(output);\n' % np_dt_to_c(self._dtype))
            self._stream.write('\n')
            self._stream.write('        std::vector<%s> expected_values{%s};\n' % (np_dt_to_c(self._dtype), print_values(data_out.reshape(-1))))
            self._stream.write('        EXPECT_EQ(output_values, expected_values);\n')
            self._stream.write('    }\n')

    def set_shape(self,shape):
        self._shape = shape

    def set_dtype(self,dtype):
        self._dtype = dtype

def main():
    assert(len(sys.argv) > 1)

    f = open(sys.argv[1], 'w')
    f.write('''\
//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// !!!!!!!!!!!!!! THIS FILE IS AUTOGENERATED OUTSIDE OF THE BUILD PROCESS !!!!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT EDIT THIS FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// DO NOT EDIT THIS FILE. If you want to add new tests, you should edit
//  test/ref_generators/generate_dyn_slice_ref.py and regenerate this file.
//
// To regenerate:
//
//   $ cd <ngraph source dir>/test
//   $ ./update_dyn_slice_reference.sh
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT EDIT THIS FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !!!!!!!!!!!!!! THIS FILE IS AUTOGENERATED OUTSIDE OF THE BUILD PROCESS !!!!!!!!!!!!!!
//
// clang-format off

#include <algorithm>
#include <cmath>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/test_tools.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, dyn_slice)
{
''')

    t = SliceTestWriter(stream=f)

    t.set_shape((4,))
    for dt in ['int32','float32']:
        t.set_dtype(dt)

        t[np.newaxis,3:0:-1]
        t[...]
        t[1:3]
        t[2]
        t[3:0:-2]
        t[3::-2]
        t[4::-2]
        t[5::-2]
        t[-9000:-8000:2]
        t[-9000:8000:2]
        t[-5:5:2]
        t[np.newaxis]
        t[np.newaxis,np.newaxis]
        t[np.newaxis,np.newaxis,...,np.newaxis]

        t.set_shape((5,))
        t[3:0:-2]
        t[0:3:2]
        t[0:4:2]
        t[0:5:2]
        t[0:6:2]
        t[0:100:2]
        t[4:0:-2]
        t[4:0:-3]
        t[3:2:1]
        t[4::-2]

    t.set_dtype('int32')
    t[80000] # error expected (shrink-axis OOB)
    t[-80000] # error expected (shrink-axis OOB)
    t[:,:] # error expected (too many indices)
    t[0:0:0] # error expected (stride==0)
    t[0:1:0] # error expected (stride==0)
    t[0:2:0] # error expected (stride==0)
    t[::0] # error expected (stride==0)

    t.set_shape((2,3,4))
    for dt in ['int32','float32']:
        t.set_dtype(dt)

        t[1,np.newaxis]
        t[-1,-1,np.newaxis]

    t.set_shape((2,4,6,8,2,2,2))
    for dt in ['int32','int64','float32','uint32']:
        t.set_dtype(dt)
        t[0:,:4,2:6:2,7:3:-2,np.newaxis,...,1]

    t.set_dtype('int32')
    t[...,...] # error expected (too many ellipses)

    f.write('''\
}
// clang-format on
''')

    f.close()

if __name__ == "__main__":
    main()
