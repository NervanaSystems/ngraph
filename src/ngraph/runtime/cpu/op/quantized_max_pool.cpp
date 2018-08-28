
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "quantized_max_pool.hpp"
#include "ngraph/function.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::QuantizedMaxPool::QuantizedMaxPool(const shared_ptr<Node>& arg,
                                       const Shape& window_shape,
                                       const Strides& window_movement_strides,
                                       const Shape& padding_below,
                                       const Shape& padding_above,
                                       const shared_ptr<Node> min,
                                       const shared_ptr<Node> max)
    : RequiresTensorViewArgs("QuantizedMaxPool", {arg, min, max})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
{
    auto& arg_shape = get_input_shape(0);
    size_t batch_size = arg_shape[0];
    size_t channel_count = arg_shape[1];
    size_t spatial_dimension_count = arg_shape.size() - 2;
    Shape input_item_virtual_shape;
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        size_t dim_size = arg_shape[1 + 1 + i];
        size_t virtual_dim_size = padding_below[i] + dim_size + padding_above[i];
        input_item_virtual_shape.push_back(virtual_dim_size);
        if (virtual_dim_size == 0)
        {
            throw ngraph_error("Max-pool input spatial dimension is zero even after padding.");
        }
    }
    Shape output_item_shape;
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_movement_strides[i] == 0)
        {
            throw ngraph_error("Max-pool window axis movement stride is zero.");
        }
        output_item_shape.push_back(ceil_div(input_item_virtual_shape[i] - window_shape[i] + 1,
                                             window_movement_strides[i]));
    }
    Shape result_shape(1 + 1 + spatial_dimension_count);
    result_shape[0] = batch_size;
    result_shape[1] = channel_count;
    copy(output_item_shape.begin(), output_item_shape.end(), result_shape.begin() + 2);
    set_value_type_checked(get_input_element_type(0), result_shape);
    add_output(element::f32, Shape{});
    add_output(element::f32, Shape{});
}

shared_ptr<Node> op::QuantizedMaxPool::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<QuantizedMaxPool>(new_args.at(0),
                                         m_window_shape,
                                         m_window_movement_strides,
                                         m_padding_below,
                                         m_padding_above,
                                         new_args.at(1),
                                         new_args.at(2));
}
