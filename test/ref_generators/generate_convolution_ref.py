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
#    filter_dilation  = (l1,...,ln)
#    below_pads       = (p1,...,pn)
#    above_pads       = (q1,...,qn)
#    image_dilation   = (g1,...,gn)
#
#    Returns:
#    output_batch     : [N ][Co][D'1]...[D'n]
#
# Where the D's are computed according to TensorFlow-style "valid" convolution rules, but *after* padding.
# See https://www.tensorflow.org/api_docs/python/tf/nn/convolution.
#
def convolution_ref(img_batch, filter, move_strides, filter_dilation, below_pads, above_pads, image_dilation):
    assert(len(img_batch.shape) == len(filter.shape))
    assert(len(img_batch.shape) > 2)
    assert(img_batch.shape[1] == filter.shape[1])
    assert(len(move_strides) == len(img_batch.shape) - 2)
    assert(len(filter_dilation) == len(img_batch.shape) - 2)
    assert(len(image_dilation) == len(img_batch.shape) - 2)

    # dilate the input batch
    new_img_shape = (np.array(img_batch.shape[2:]) - 1) * image_dilation + 1
    new_img_batch_shape = list(np.array(img_batch.shape[:2])) + list(new_img_shape)
    new_img_batch = np.zeros(new_img_batch_shape)

    for n in range(0, new_img_batch_shape[0]) :
        for c in range(0, new_img_batch_shape[1]) :
            if new_img_batch.ndim == 4:
                new_img_batch[n, c, 0::image_dilation[0], 0::image_dilation[1]] = img_batch[n][c]
            elif new_img_batch.ndim == 5:
                new_img_batch[n, c, 0::image_dilation[0], 0::image_dilation[1], 0::image_dilation[2]] = img_batch[n][c]
            elif new_img_batch.ndim == 6:
                new_img_batch[n, c, 0::image_dilation[0], 0::image_dilation[1], 0::image_dilation[2], 0::image_dilation[3]] = img_batch[n][c]
            else:
                assert(False)

    img_batch = new_img_batch

    # Pad the input batch.
    below_pads = (0,0) + below_pads  # Have to add values for the image and channel dims.
    above_pads = (0,0) + above_pads  # Have to add values for the image and channel dims.
    img_batch = np.pad(img_batch, zip(below_pads,above_pads), mode='constant', constant_values=0)

    img_count = img_batch.shape[0]                # N
    ci_count = img_batch.shape[1]                 # Ci
    co_count = filter.shape[0]                    # Co
    input_img_shape = list(img_batch.shape[2:])   # D1, ..., Dn
    window_virtual_shape = list(filter.shape[2:]) # W1, ..., Wn

    # This is not used in computation but we will calculate it for a check to make sure the window fits.
    window_physical_shape = []
    for (d_in,d_virt,dil) in zip(input_img_shape,window_virtual_shape,filter_dilation):
        d_phys = (d_virt - 1) * dil + 1
        assert(d_phys <= input_img_shape)
        window_physical_shape.append(d_phys)

    output_img_shape = []  # D'1,...,D'n
    for (d_in,d_win,dil,mov) in zip (input_img_shape,window_virtual_shape,filter_dilation,move_strides):
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
            input_index = (img,ci) + (tuple_plus(tuple_times(output_pos,move_strides),tuple_times(filter_pos,filter_dilation)))

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
            result = ('%.1000g' % x)
            first = False
        else:
            result = result + (',%.1000g' % x)
    return result

def emit_test(t,f):
    test_name, input_batch_data, filter_data, move_strides, filter_dilation, below_pads, above_pads, image_dilation = t

    print ("Generating convolution test '%s'..." % test_name)

    output_batch_data = convolution_ref(input_batch_data,filter_data,move_strides,filter_dilation,below_pads,above_pads,image_dilation)

    template = '''
TEST (${BACKEND_NAME}, %s)
{
    auto shape_a = Shape{%s};
    auto A = make_shared<op::Parameter>(element::f64, shape_a);
    auto shape_b = Shape{%s};
    auto B = make_shared<op::Parameter>(element::f64, shape_b);
    auto shape_r = Shape{%s};
    auto f = make_shared<Function>(
        make_shared<op::Convolution>(A, B, 
                                     Strides{%s},  // move_strides
                                     Strides{%s},  // filter_dilation
                                     Padding{%s},  // below_pads
                                     Padding{%s},  // above_pads
                                     Strides{%s}), // image_dilation
        op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::f64, shape_a);
    copy_data(a, vector<double>{%s});
    auto b = backend->make_primary_tensor_view(element::f64, shape_b);
    copy_data(b, vector<double>{%s});
    auto result = backend->make_primary_tensor_view(element::f64, shape_r);

    vector<double> expected_result{%s};

    cf->call({a, b}, {result});
    EXPECT_TRUE(all_close_d(vector<double>{expected_result},
                            result->get_vector<double>()));
}
'''
    f.write (template % (test_name,
                         shape_str(input_batch_data.shape),
                         shape_str(filter_data.shape),
                         shape_str(output_batch_data.shape),
                         shape_str(move_strides),
                         shape_str(filter_dilation),
                         shape_str(below_pads),
                         shape_str(above_pads),
                         shape_str(image_dilation),
                         data_str(input_batch_data),
                         data_str(filter_data),
                         data_str(output_batch_data)));

#                                                                                                                          filter                           image
#         test name                                input image batch              filters                        stride    dilation  below-pads  above-pads dilation
tests = [
         ("convolution_2d_1image",                 shaped_linspace((1,1,3,5)),    shaped_linspace((2,1,2,2)),    (1,1),    (1,1),    (0,0),      (0,0),     (1,1)),
         ("convolution_2d_1image_padded_1_1x1_1",  shaped_linspace((1,1,3,5)),    shaped_linspace((2,1,2,2)),    (1,1),    (1,1),    (1,1),      (1,1),     (1,1)),
         ("convolution_2d_1image_padded_2_3x4_5",  shaped_linspace((1,1,3,5)),    shaped_linspace((2,1,2,2)),    (1,1),    (1,1),    (2,3),      (4,5),     (1,1)),
         ("convolution_2d_2images",                shaped_linspace((2,1,3,5)),    shaped_linspace((2,1,2,2)),    (1,1),    (1,1),    (0,0),      (0,0),     (1,1)),
         ("convolution_2d_2images_strided",        shaped_linspace((2,1,3,5)),    shaped_linspace((2,1,2,2)),    (2,2),    (1,1),    (0,0),      (0,0),     (1,1)),
         ("convolution_2d_2images_strided_padded", shaped_linspace((2,1,3,5)),    shaped_linspace((2,1,2,2)),    (2,2),    (1,1),    (4,2),      (5,7),     (1,1)),
         ("convolution_2d_2images_strided_padded_same",
                                                   shaped_linspace((2,1,3,5)),    shaped_linspace((2,1,2,2)),    (2,2),    (1,1),    (2,2),      (2,2),     (1,1)),
         ("convolution_2d_2images_dilated",        shaped_linspace((2,1,3,5)),    shaped_linspace((2,1,2,2)),    (1,1),    (2,2),    (0,0),      (0,0),     (1,1)),
         ("convolution_2d_2images_dilated_padded", shaped_linspace((2,1,3,5)),    shaped_linspace((2,1,2,2)),    (1,1),    (2,2),    (4,2),      (5,7),     (1,1)),
         ("convolution_3d_2images",                shaped_linspace((2,1,3,5,8)),  shaped_linspace((2,1,2,2,3)),  (1,1,1),  (1,1,1),  (0,0,0),    (0,0,0),   (1,1,1)),
         ("convolution_4d_2images",                shaped_linspace((2,1,3,5,8,7)),shaped_linspace((2,1,2,2,3,1)),(1,1,1,1),(1,1,1,1),(0,0,0,0),  (0,0,0,0), (1,1,1,1)),
         ("convolution_4d_4images",                shaped_linspace((4,3,3,5,8,7)),shaped_linspace((4,3,2,2,3,1)),(1,1,1,1),(1,1,1,1),(0,0,0,0),  (0,0,0,0), (1,1,1,1)),
         ("convolution_4d_4images_strided",        shaped_linspace((4,3,3,5,8,7)),shaped_linspace((4,3,2,2,3,1)),(2,1,3,2),(1,1,1,1),(0,0,0,0),  (0,0,0,0), (1,1,1,1)),
         ("convolution_4d_4images_dilated",        shaped_linspace((4,3,3,5,8,7)),shaped_linspace((4,3,2,2,3,1)),(1,1,1,1),(2,1,3,2),(0,0,0,0),  (0,0,0,0), (1,1,1,1)),
         ("convolution_4d_4images_strided_dilated",shaped_linspace((4,3,8,8,8,8)),shaped_linspace((4,3,2,2,3,1)),(3,2,2,3),(2,1,3,2),(0,0,0,0),  (0,0,0,0), (1,1,1,1)),
         ("convolution_4d_4images_strided_dilated_padded",
                                                   shaped_linspace((4,3,8,8,8,8)),shaped_linspace((4,3,2,2,3,1)),(3,2,2,3),(2,1,3,2),(2,4,6,8),  (1,3,5,7), (1,1,1,1)),
         ("convolution_4d_4images_strided_dilated_padded_same",
                                                   shaped_linspace((4,3,8,8,8,8)),shaped_linspace((4,3,2,2,3,1)),(3,2,2,3),(2,1,3,2),(3,3,3,3),  (3,3,3,3), (1,1,1,1)),
         ("convolution_2d_1image_1o1i_img_dilated",shaped_linspace((1,1,3,5)),    shaped_linspace((1,1,2,2)),    (1,1),    (1,1),    (0,0),      (0,0),     (2,2)),
         ("convolution_2d_1image_2o1i_img_dilated",shaped_linspace((1,1,3,5)),    shaped_linspace((2,1,2,2)),    (1,1),    (1,1),    (0,0),      (0,0),     (2,2)),
         ("convolution_2d_1image_2o2i_img_dilated",shaped_linspace((1,2,3,5)),    shaped_linspace((2,2,2,2)),    (1,1),    (1,1),    (0,0),      (0,0),     (2,2)),
         ("convolution_2d_1image_5o3i_img_dilated",shaped_linspace((1,3,3,5)),    shaped_linspace((5,3,2,2)),    (1,1),    (1,1),    (0,0),      (0,0),     (2,2)),
         ("convolution_2d_8image_5o3i_img_dilated",shaped_linspace((8,3,3,5)),    shaped_linspace((5,3,2,2)),    (1,1),    (1,1),    (0,0),      (0,0),     (2,2)),
         ("convolution_2d_8image_large_5o3i_img_dilated",
                                                   shaped_linspace((8,3,16,16)),  shaped_linspace((5,3,2,2)),    (1,1),    (1,1),    (0,0),      (0,0),     (2,2)),
         ("convolution_2d_8image_large_5o3i_uneven_filter_img_dilated",
                                                   shaped_linspace((8,3,16,16)),  shaped_linspace((5,3,2,3)),    (1,1),    (1,1),    (0,0),      (0,0),     (2,2)),
         ("convolution_2d_8image_large_5o3i_uneven_filter_uneven_img_dilation_img_dilated",
                                                   shaped_linspace((8,3,16,16)),  shaped_linspace((5,3,2,3)),    (1,1),    (1,1),    (0,0),      (0,0),     (2,3)),
         ("convolution_3d_2image_large_5o3i_uneven_filter_uneven_img_dilation_img_dilated",
                                                   shaped_linspace((2,3,8,8,8)),  shaped_linspace((5,3,2,3,4)),  (1,1,1),  (1,1,1),  (0,0,0),    (0,0,0),   (2,3,2)),
         ("convolution_3d_1image_large_5o3i_padded_uneven_filter_uneven_img_dilation_img_dilated",
                                                   shaped_linspace((1,3,8,8,8)),  shaped_linspace((5,3,2,3,4)),  (1,1,1),  (1,1,1),  (2,1,2),    (1,2,3),   (2,3,2)),
         ("convolution_3d_2image_large_5o3i_padded_strided_uneven_filter_uneven_img_dilation_img_dilated",
                                                   shaped_linspace((2,3,8,8,8)),  shaped_linspace((5,3,2,3,4)),  (2,3,2),  (1,1,1),  (2,1,2),    (1,2,3),   (2,3,2)),
         ("convolution_3d_2image_large_5o3i_padded_strided_uneven_filter_uneven_img_dilation_filter_dilated_img_dilated",
                                                   shaped_linspace((2,3,8,8,8)),  shaped_linspace((5,3,2,3,4)),  (2,3,2),  (3,2,2),  (2,1,2),    (1,2,3),   (2,3,2)),
        ]

def main():
    assert(len(sys.argv)>1)

    f = open(sys.argv[1],'w')
    f.write('''
// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------
//
// !!!!!!!!!!!!!!THIS FILE IS AUTOGENERATED OUTSIDE OF THE BUILD PROCESS!!!!!!!!!!!!!!
//
// It takes quite awhile to compute the results, so doing it at cmake time is not a good option.
//
// If you want to add new tests, you should edit test/ref_generators/generate_convolution_ref.py
// and regenerate this file.
//
// To regenerate (NOTE: this script will run apply-code-format.sh and reformat all source files
// in your tree):
//
//   $ cd <ngraph source dir>/test
//   $ ./update_reference.sh
//
// !!!!!!!!!!!!!!THIS FILE IS AUTOGENERATED OUTSIDE OF THE BUILD PROCESS!!!!!!!!!!!!!!
//

#include <cmath>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

static bool all_close_d(const std::vector<double>& a,
                        const std::vector<double>& b,
                        double rtol = 1e-5,
                        double atol = 1e-8)
{
    assert(a.size() == b.size());

    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i]))
        {
            return false;
        }
    }
    return true;
}
''')
    for t in tests:
        emit_test(t,f)
    f.close()

if __name__ == "__main__":
    main()
