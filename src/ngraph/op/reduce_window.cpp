//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/reduce_window.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::ReduceWindow::ReduceWindow(const shared_ptr<Node>& arg_reductee,
                               const shared_ptr<Node>& arg_init,
                               const shared_ptr<Function>& reduction_function,
                               const Shape& window_shape,
                               const Strides& window_movement_strides)
    : Op("ReduceWindow", check_single_output_args({arg_reductee, arg_init}))
    , m_reduction_function(reduction_function)
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
{
    constructor_validate_and_infer_types();

    auto& input_reductee = get_inputs().at(0);
    auto& input_init = get_inputs().at(1);
    auto input_reductee_shape = input_reductee.get_shape();
    auto input_init_shape = input_init.get_shape();

    if (input_init.get_shape().size() != 0)
    {
        throw ngraph_error("Argument for initial value is not a scalar");
    }

    if (input_init.get_element_type() != input_reductee.get_element_type())
    {
        throw ngraph_error("Element types for reductee and initial values do not match");
    }

    if (input_reductee_shape.size() != window_shape.size())
    {
        throw ngraph_error("Window shape has different rank from input tensor");
    }

    if (input_reductee_shape.size() != window_movement_strides.size())
    {
        throw ngraph_error("Window movement strides have different rank from input tensor");
    }

    for (size_t s : window_shape)
    {
        if (s == 0)
        {
            throw ngraph_error("Window shape has a zero-length axis");
        }
    }

    for (size_t s : window_movement_strides)
    {
        if (s == 0)
        {
            throw ngraph_error("Window movement stride for some axis is zero");
        }
    }

    for (size_t i = 0; i < input_reductee_shape.size(); i++)
    {
        if (window_shape[i] > input_reductee_shape[i])
        {
            throw ngraph_error("Reduction window is bigger than input");
        }
    }

    auto f_params = m_reduction_function->get_parameters();

    if (f_params.size() != 2)
    {
        throw ngraph_error("Reduction function has wrong number of parameters (should be two)");
    }

    if (f_params.at(0)->get_element_type() != arg_init->get_element_type())
    {
        throw ngraph_error("Parameter 0 of reduction function has wrong element type");
    }
    if (f_params.at(1)->get_element_type() != arg_init->get_element_type())
    {
        throw ngraph_error("Parameter 1 of reduction function has wrong element type");
    }

    if (f_params.at(0)->get_shape() != Shape{})
    {
        throw ngraph_error("Parameter 0 of reduction function is not a scalar");
    }
    if (f_params.at(1)->get_shape() != Shape{})
    {
        throw ngraph_error("Parameter 1 of reduction function is not a scalar");
    }

    if (m_reduction_function->get_output_size() > 1)
    {
        throw ngraph_error("Single-output reduction function was expected");
    }

    if (m_reduction_function->get_output_element_type(0) != arg_init->get_element_type())
    {
        throw ngraph_error("Return element type from reduction function does not match expected");
    }
    if (m_reduction_function->get_output_shape(0) != Shape{})
    {
        throw ngraph_error("Return shape from reduction function is not a scalar");
    }

    Shape result_shape;

    for (size_t i = 0; i < input_reductee_shape.size(); i++)
    {
        result_shape.push_back(
            ceil_div(input_reductee_shape[i] - window_shape[i] + 1, window_movement_strides[i]));
    }

    set_output_type(0, input_reductee.get_element_type(), result_shape);
}

shared_ptr<Node> op::ReduceWindow::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    auto node = make_shared<ReduceWindow>(new_args.at(0),
                                          new_args.at(1),
                                          m_reduction_function,
                                          m_window_shape,
                                          m_window_movement_strides);
    node->m_reduction_function = clone_function(*m_reduction_function);
    return node;
}
