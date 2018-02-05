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

#include "ngraph/ops/max_pool.hpp"
#include "ngraph/function.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/select_and_scatter.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::MaxPool::MaxPool(const std::shared_ptr<Node>& arg,
                     const Shape& window_shape,
                     const Strides& window_movement_strides)
    : RequiresTensorViewArgs("MaxPool", {arg})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
{
    auto& arg_shape = get_inputs().at(0).get_shape();

    //
    // Make sure arg: NCDi for some Di of rank>0, N != 0, C != 0.
    //
    if (arg_shape.size() < 3)
    {
        throw ngraph_error(
            "Max pool data batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one spatial dimension).");
    }

    size_t batch_size = arg_shape[0];
    if (batch_size == 0)
    {
        throw ngraph_error("Max pool data batch size is zero.");
    }

    size_t channel_count = arg_shape[1];
    if (channel_count == 0)
    {
        throw ngraph_error("Max pool requires at least one feature channel.");
    }

    size_t spatial_dimension_count = arg_shape.size() - 2;

    //
    // Make sure window shape and movement strides have same rank as Di.
    //
    if (window_shape.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max pool window shape rank does not match number of spatial dimensions.");
    }

    if (window_movement_strides.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            "Max pool window movement stride rank does not match number of spatial dimensions.");
    }

    //
    // Extract input item shape Di and make sure all dimensions are larger than 0.
    //
    Shape input_spatial_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        input_spatial_shape.push_back(arg_shape[1 + 1 + i]);

        if (input_spatial_shape[i] == 0)
        {
            throw ngraph_error("Max pool input spatial dimension is zero.");
        }
    }

    //
    // Make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_shape[i] == 0)
        {
            throw ngraph_error("Max pool window shape has a zero-length axis.");
        }
    }

    //
    // Make the max pooling window fits within the spatial dimensions.
    //
    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_shape[i] > input_spatial_shape[i])
        {
            throw ngraph_error("Max pool window shape is larger than the spatial dimensions.");
        }
    }

    //
    // Compute output item shape Do, checking at the same time that all window movement strides are larger than 0.
    //
    Shape output_spatial_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_movement_strides[i] == 0)
        {
            throw ngraph_error("Max pool window axis movement stride is zero.");
        }
        output_spatial_shape.push_back(
            ceil_div(input_spatial_shape[i] - window_shape[i] + 1, window_movement_strides[i]));
    }

    //
    // Construct result shape: NCDo.
    //
    Shape result_shape(1 + 1 + spatial_dimension_count);
    result_shape[0] = batch_size;
    result_shape[1] = channel_count;
    std::copy(output_spatial_shape.begin(), output_spatial_shape.end(), result_shape.begin() + 2);

    set_value_type_checked(get_inputs().at(0).get_element_type(), result_shape);
}

static Strides default_strides(const std::shared_ptr<Node>& arg)
{
    if (arg->get_outputs().size() != 1)
    {
        throw ngraph_error("Max pool data batch argument must have exactly one output");
    }

    auto& arg_shape = arg->get_outputs().at(0).get_shape();
    if (arg_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Max pool data batch input must have rank of at least 3 (one batch axis, one "
            "channel axis, at least one spatial dimension).");
    }
    return Strides(arg_shape.size() - 2, 1);
}

op::MaxPool::MaxPool(const std::shared_ptr<Node>& arg, const Shape& window_shape)
    : MaxPool(arg, window_shape, default_strides(arg))
{
}

void op::MaxPool::generate_adjoints(autodiff::Adjoints& adjoints,
                                    const std::shared_ptr<Node>& delta)
{
    auto shape_sel_a = Shape{};
    auto etype = delta->get_element_type();

    //Select Max
    auto SEL_A = make_shared<op::Parameter>(etype, shape_sel_a);
    auto shape_sel_b = Shape{};
    auto SEL_B = make_shared<op::Parameter>(etype, shape_sel_b);
    auto sel_f = std::make_shared<Function>(std::make_shared<op::Greater>(SEL_A, SEL_B),
                                            op::Parameters{SEL_A, SEL_B});

    //Update Cell
    auto shape_scatter_a = Shape{};
    auto SCATTER_A = make_shared<op::Parameter>(etype, shape_scatter_a);
    auto shape_scatter_b = Shape{};
    auto SCATTER_B = make_shared<op::Parameter>(etype, shape_scatter_b);
    auto scatter_f =
        make_shared<Function>(SCATTER_A + SCATTER_B, op::Parameters{SCATTER_A, SCATTER_B});

    auto operand = get_input_op(0);
    auto init_value =
        std::make_shared<op::Constant>(etype, Shape{}, std::vector<std::string>({"0"}));

    Strides strides{1, 1};
    strides.push_back(m_window_movement_strides.at(0));
    strides.push_back(m_window_movement_strides.at(1));

    Shape shape{1, 1};
    shape.push_back(m_window_shape.at(0));
    shape.push_back(m_window_shape.at(1));

    auto sas = std::make_shared<op::SelectAndScatter>(
        operand, delta, init_value, sel_f, scatter_f, shape, strides);
    adjoints.add_delta(operand, sas);
}
