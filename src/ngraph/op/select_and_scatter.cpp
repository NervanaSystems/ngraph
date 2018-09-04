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

#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::SelectAndScatter::SelectAndScatter(const shared_ptr<Node>& arg_selectee,
                                       const shared_ptr<Node>& arg_source,
                                       const shared_ptr<Node>& arg_init,
                                       const shared_ptr<Function>& selection_function,
                                       const shared_ptr<Function>& scatter_function,
                                       const Shape& window_shape,
                                       const Strides& window_movement_strides)
    : Op("SelectAndScatter", check_single_output_args({arg_selectee, arg_source, arg_init}))
    , m_selection_function(selection_function)
    , m_scatter_function(scatter_function)
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
{
    constructor_validate_and_infer_types();

    auto& input_selectee = get_inputs().at(0);
    auto& input_source = get_inputs().at(1);
    auto& input_init = get_inputs().at(2);
    auto input_selectee_shape = input_selectee.get_shape();
    auto input_source_shape = input_source.get_shape();
    auto input_init_shape = input_init.get_shape();
    auto& input_selectee_element_type = input_selectee.get_element_type();
    auto& input_source_element_type = input_source.get_element_type();
    auto& input_init_element_type = input_init.get_element_type();

    //
    // Make sure the initial value is a scalar.
    //
    if (input_init_shape.size() != 0)
    {
        throw ngraph_error("Argument for initial value is not a scalar");
    }

    //
    // Make sure input element types all match.
    //
    if (input_init_element_type != input_selectee_element_type)
    {
        throw ngraph_error("Element types for selectee and initial values do not match");
    }

    if (input_source_element_type != input_selectee_element_type)
    {
        throw ngraph_error("Element types for selectee and source tensors do not match");
    }

    //
    // Check that the window shape and strides have the right rank.
    //
    if (input_selectee_shape.size() != window_shape.size())
    {
        throw ngraph_error("Window shape has different rank from selectee tensor");
    }

    if (input_selectee_shape.size() != window_movement_strides.size())
    {
        throw ngraph_error("Window movement strides have different rank from selectee tensor");
    }

    //
    // Check for zero-length window axes or strides.
    //
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

    //
    // Check that the window is not bigger than the selectee tensor.
    //
    for (size_t i = 0; i < input_selectee_shape.size(); i++)
    {
        if (window_shape[i] > input_selectee_shape[i])
        {
            throw ngraph_error("Reduction window is bigger than selectee tensor");
        }
    }

    //
    // The expected shape of the source tensor is the same as the shape of the output
    // we would get if we window-reduced the selectee; in other words, this logic is
    // the same as the logic for computing the output shape of reduce-window.
    //
    Shape expected_source_shape;

    for (size_t i = 0; i < input_selectee_shape.size(); i++)
    {
        expected_source_shape.push_back(
            ceil_div(input_selectee_shape[i] - window_shape[i] + 1, window_movement_strides[i]));
    }

    if (input_source_shape != expected_source_shape)
    {
        throw ngraph_error("Source tensor does not have expected shape");
    }

    //
    // Check the type signature of the selection function. Should be T -> T -> Bool.
    //
    auto selection_function_params = m_selection_function->get_parameters();

    if (selection_function_params.size() != 2)
    {
        throw ngraph_error("Selection function has wrong number of parameters (should be two)");
    }

    if (selection_function_params.at(0)->get_element_type() != arg_init->get_element_type())
    {
        throw ngraph_error("Parameter 0 of selection function has wrong element type");
    }
    if (selection_function_params.at(1)->get_element_type() != arg_init->get_element_type())
    {
        throw ngraph_error("Parameter 1 of selection function has wrong element type");
    }
    if (selection_function_params.at(0)->get_shape() != Shape{})
    {
        throw ngraph_error("Parameter 0 of selection function is not a scalar");
    }
    if (selection_function_params.at(1)->get_shape() != Shape{})
    {
        throw ngraph_error("Parameter 1 of selection function is not a scalar");
    }

    if (m_selection_function->get_output_size() > 1)
    {
        throw ngraph_error("Single-output selection function was expected");
    }

    if (m_selection_function->get_output_element_type(0) != element::boolean)
    {
        throw ngraph_error("Return element type from selection function is not boolean");
    }
    if (m_selection_function->get_output_shape(0) != Shape{})
    {
        throw ngraph_error("Return shape from selection function is not a scalar");
    }

    //
    // Check the type signature of the scatter function. Should be T -> T -> T.
    //
    auto scatter_function_params = m_scatter_function->get_parameters();

    if (scatter_function_params.size() != 2)
    {
        throw ngraph_error("Scatter function has wrong number of parameters (should be two)");
    }

    if (scatter_function_params.at(0)->get_element_type() != arg_init->get_element_type())
    {
        throw ngraph_error("Parameter 0 of scatter function has wrong element type");
    }
    if (scatter_function_params.at(1)->get_element_type() != arg_init->get_element_type())
    {
        throw ngraph_error("Parameter 1 of scatter function has wrong element type");
    }

    if (scatter_function_params.at(0)->get_shape() != Shape{})
    {
        throw ngraph_error("Parameter 0 of scatter function is not a scalar");
    }
    if (scatter_function_params.at(1)->get_shape() != Shape{})
    {
        throw ngraph_error("Parameter 1 of scatter function is not a scalar");
    }

    if (m_scatter_function->get_output_size() > 1)
    {
        throw ngraph_error("Single-output scatter function was expected");
    }

    if (m_scatter_function->get_output_element_type(0) != arg_init->get_element_type())
    {
        throw ngraph_error(
            "Return element type from scatter function does not match the init value type");
    }
    if (m_scatter_function->get_output_shape(0) != Shape{})
    {
        throw ngraph_error("Return shape from scatter function is not a scalar");
    }

    //
    // Result type is the same element type and shape as the selectee.
    //
    set_output_type(0, input_selectee_element_type, input_selectee_shape);
}

shared_ptr<Node> op::SelectAndScatter::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    auto node = make_shared<SelectAndScatter>(new_args.at(0),
                                              new_args.at(1),
                                              new_args.at(2),
                                              m_selection_function,
                                              m_scatter_function,
                                              m_window_shape,
                                              m_window_movement_strides);
    node->m_selection_function = clone_function(*m_selection_function);
    node->m_scatter_function = clone_function(*m_scatter_function);
    return node;
}
