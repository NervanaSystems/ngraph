#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
# ----------------------------------------------------------------------------

import sys
import numpy as np
import math
from operator import mul

# Imposes the shape on the given 1-D array to produce a C-style-indexed n-D array.
def shaped_from_flat(shape,flat):
    total_elems = reduce(mul,shape)

    assert(len(flat) == total_elems)

    arr = np.array(flat)
    arr.shape = shape

    return arr

# Creates a linspaced array from 1 to n where n is the number of elements in the shape, then
# imposes the shape on the array to produce a C-style-indexed n-D array.
def shaped_linspace(shape):
    total_elems = reduce(mul,shape)

    flat = np.linspace(1,total_elems,total_elems)
    
    return shaped_from_flat(shape,flat)

# Elementwise addition on tuples.
def tuple_plus(t1,t2):
    assert(len(t1) == len(t2))

    res = ()

    for (x,y) in zip(list(t1),list(t2)):
        res = res + (x+y,)

    return res

# Elementwise multiplication on tuples.
def tuple_times(t1,t2):
    assert(len(t1) == len(t2))

    res = ()

    for (x,y) in zip(list(t1),list(t2)):
        res = res + (x*y,)

    return res

#
# Convolution reference
#
#    Arguments:
#    img_batch        : [N ][Ci][D1]...[Dn], n > 0
#    filter           : [Co][Ci][W1]...[Wn]
#    move_strides     = (s1,...,sn)
#    dilation_strides = (l1,...,ln)
#
#    Returns:
#    output_batch     : [N ][Co][D'1]...[D'n]
#
# Where the D's are computed according to TensorFlow-style "valid" convolution rules.
# See https://www.tensorflow.org/api_docs/python/tf/nn/convolution.
#
def convolution_ref(img_batch, filter, move_strides, dilation_strides):
    assert(len(img_batch.shape) == len(filter.shape))
    assert(len(img_batch.shape) > 2)
    assert(img_batch.shape[1] == filter.shape[1])
    assert(len(move_strides) == len(img_batch.shape) - 2)
    assert(len(dilation_strides) == len(img_batch.shape) - 2)

    img_count = img_batch.shape[0]                # N
    ci_count = img_batch.shape[1]                 # Ci
    co_count = filter.shape[0]                    # Co
    input_img_shape = list(img_batch.shape[2:])   # D1, ..., Dn
    window_virtual_shape = list(filter.shape[2:]) # W1, ..., Wn

    # This is not used in computation but we will calculate it for a check to make sure the window fits.
    window_physical_shape = []
    for (d_in,d_virt,dil) in zip(input_img_shape,window_virtual_shape,dilation_strides):
        d_phys = (d_virt - 1) * dil + 1
        assert(d_phys <= input_img_shape)
        window_physical_shape.append(d_phys)

    output_img_shape = []  # D'1,...,D'n
    for (d_in,d_win,dil,mov) in zip (input_img_shape,window_virtual_shape,dilation_strides,move_strides):
        d_out = int(math.ceil((float(d_in) - (float(d_win) - 1.0) * float(dil))/float(mov))) # Formula is taken from TF's definition for VALID convolution.
        assert(d_out > 0)
        output_img_shape.append(d_out)

    output_shape = [img_count,co_count]+output_img_shape # N,Co,D'1,...,D'n
    output_batch = np.zeros(output_shape)

    # Walk over the output batch space.
    output_it = np.nditer(output_batch, flags=['multi_index'])
    while not output_it.finished:
        # Break up the output coordinate to figure out where we are in terms of image index, output channel, and image shape position.
        output_index = output_it.multi_index
        img, co, output_pos = output_index[0], output_index[1], output_index[2:]

        # Walk over the filter for the current output channel.
        filter_it = np.nditer(filter[co], flags=['multi_index'])
        while not filter_it.finished:
            # Break up the filter coordinate to figure out where we are in terms of input channel and filter shape position.
            filter_index = filter_it.multi_index
            ci, filter_pos = filter_index[0], filter_index[1:]

            # Build up the coordinate within the space N,Ci,D1,...,Dn that we need to read from in the input batch.
            input_index = (img,ci) + (tuple_plus(tuple_times(output_pos,move_strides),tuple_times(filter_pos,dilation_strides)))

            # Add to the sum-of-products.
            output_batch[output_index] = output_batch[output_index] + filter[(co,) + filter_index] * img_batch[input_index]

            filter_it.iternext()

        output_it.iternext()

    return output_batch

def shape_str(shape):
    result = ''
    first = True
    for d in shape:
        if first:
            result = ('%d' % d)
            first = False
        else:
            result = result + (',%d' % d)
    return result

def data_str(data):
    result = ''
    first = True
    for x in np.nditer(data):
        if first:
            result = ('%f' % x)
            first = False
        else:
            result = result + (',%f' % x)
    return result

def emit_test(t,f):
    test_name, input_batch_data, filter_data, move_strides, dilation_strides = t

    print ("Generating convolution test '%s'..." % test_name)

    output_batch_data = convolution_ref(input_batch_data,filter_data,move_strides,dilation_strides)

    template = '''
TEST (CONV_TEST_BACKEND, %s)
{
    auto shape_a = Shape{%s};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{%s};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{%s};
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Convolution>(A, B, Strides{%s}, Strides{%s}), result_type, op::Parameters{A, B});

    auto manager = runtime::Manager::get(CONV_TEST_BACKEND_STR);
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{%s});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{%s});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    EXPECT_TRUE(test::all_close(vector<float>{%s},
                                result->get_vector<float>()));
}
'''
    f.write (template % (test_name,
                         shape_str(input_batch_data.shape),
                         shape_str(filter_data.shape),
                         shape_str(output_batch_data.shape),
                         shape_str(move_strides),
                         shape_str(dilation_strides),
                         data_str(input_batch_data),
                         data_str(filter_data),
                         data_str(output_batch_data)));

#         test name                                input image batch                  filters                         stride    dilation
tests = [
         ("convolution_2d_1image",                 shaped_linspace((1,1,3,5)),        shaped_linspace((2,1,2,2)),     (1,1),    (1,1)),
         ("convolution_2d_2images",                shaped_linspace((2,1,3,5)),        shaped_linspace((2,1,2,2)),     (1,1),    (1,1)),
         ("convolution_2d_2images_strided",        shaped_linspace((2,1,3,5)),        shaped_linspace((2,1,2,2)),     (2,2),    (1,1)),
         ("convolution_2d_2images_dilated",        shaped_linspace((2,1,3,5)),        shaped_linspace((2,1,2,2)),     (1,1),    (2,2)),
         ("convolution_3d_2images",                shaped_linspace((2,1,3,5,8)),      shaped_linspace((2,1,2,2,3)),   (1,1,1),  (1,1,1)),
         ("convolution_4d_2images",                shaped_linspace((2,1,3,5,8,7)),    shaped_linspace((2,1,2,2,3,1)), (1,1,1,1),(1,1,1,1)),
         ("convolution_4d_16images",               shaped_linspace((16,3,3,5,8,7)),   shaped_linspace((16,3,2,2,3,1)),(1,1,1,1),(1,1,1,1)),
         ("convolution_4d_16images_strided",       shaped_linspace((16,3,3,5,8,7)),   shaped_linspace((16,3,2,2,3,1)),(2,1,3,2),(1,1,1,1)),
         ("convolution_4d_16images_dilated",       shaped_linspace((16,3,3,5,8,7)),   shaped_linspace((16,3,2,2,3,1)),(1,1,1,1),(2,1,3,2)),
         ("convolution_4d_4images_strided_dilated",shaped_linspace((4,3,16,16,16,16)),shaped_linspace((4,3,2,2,3,1)), (3,2,2,3),(2,1,3,2)),
        ]

def main():
    assert(len(sys.argv)>1)

    f = open(sys.argv[1],'w')
    for t in tests:
        emit_test(t,f)
    f.close()

if __name__ == "__main__":
    main()
