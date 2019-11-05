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

#include "ngraph/op/gather.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"

#include <limits>

using namespace std;
using namespace ngraph;

static const int PARAMS = 0;
static const int INDICES = 1;
static const int AXIS = 2;

static const int64_t AXIS_NOT_SET_VALUE = std::numeric_limits<int64_t>::max();

constexpr NodeTypeInfo op::v0::Gather::type_info;

op::v0::Gather::Gather(const Output<Node>& params, const Output<Node>& indices, size_t axis)
    : Op({params, indices})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Gather::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::Gather>(new_args.at(PARAMS), new_args.at(INDICES), m_axis);
}

void op::v0::Gather::validate_and_infer_types()
{
    element::Type result_et = get_input_element_type(PARAMS);
    element::Type indices_et = get_input_element_type(INDICES);

    const PartialShape& params_shape = get_input_partial_shape(PARAMS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32");

    // params rank must be at least (axis + 1)
    // indices value must be in range [0, params.shape[axis]).
    // output rank is rank(params) + rank(indices) - 1
    NODE_VALIDATION_CHECK(this,
                          params_shape.rank().is_dynamic() ||
                              static_cast<size_t>(params_shape.rank()) >
                                  static_cast<size_t>(m_axis),
                          "params rank is expected to be at least axis + 1");

    PartialShape result_shape;
    if (params_shape.rank().is_static() && indices_shape.rank().is_static())
    {
        std::vector<Dimension> result_dims(static_cast<size_t>(params_shape.rank()) +
                                           static_cast<size_t>(indices_shape.rank()) - 1);
        size_t i = 0;
        for (; i < static_cast<size_t>(m_axis); i++)
        {
            result_dims[i] = params_shape[i];
        }
        for (size_t j = 0; j < static_cast<size_t>(indices_shape.rank()); i++, j++)
        {
            result_dims[i] = indices_shape[j];
        }
        for (size_t j = static_cast<size_t>(m_axis) + 1;
             j < static_cast<size_t>(params_shape.rank());
             i++, j++)
        {
            result_dims[i] = params_shape[j];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

void op::v0::Gather::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                       const NodeVector& /* deltas */)
{
    throw ngraph_error("Not yet implemented");
}

constexpr NodeTypeInfo op::v1::Gather::type_info;

op::v1::Gather::Gather(const Output<Node>& params,
                       const Output<Node>& indices,
                       const Output<Node>& axes)
    : Op({params, indices, axes})
{
    constructor_validate_and_infer_types();
}

void op::v1::Gather::validate_and_infer_types()
{
    const auto& input_rank = get_input_partial_shape(PARAMS).rank();
    const auto& axis_shape = get_input_partial_shape(AXIS);
    const auto& axis_rank = axis_shape.rank();

    if (axis_rank.is_static() && axis_shape.is_static())
    {
        const auto axis_is_scalar = static_cast<size_t>(axis_rank) == 0;
        const auto axis_has_one_elem =
            static_cast<size_t>(axis_rank) == 1 && static_cast<size_t>(axis_shape[0]) == 1;
        NODE_VALIDATION_CHECK(this,
                              axis_is_scalar || axis_has_one_elem,
                              "Axes input must be scalar or have 1 element (shape: ",
                              axis_shape,
                              ").");
    }

    auto axis = get_axis();
    if (input_rank.is_static() && axis != AXIS_NOT_SET_VALUE)
    {
        NODE_VALIDATION_CHECK(this,
                              axis >= 0 && axis < static_cast<size_t>(input_rank),
                              "The axis must => 0 and <= input_rank (axis: ",
                              axis,
                              ").");
    }

    element::Type result_et = get_input_element_type(PARAMS);
    element::Type indices_et = get_input_element_type(INDICES);

    const PartialShape& params_shape = get_input_partial_shape(PARAMS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    PartialShape result_shape;
    if (params_shape.rank().is_static() && indices_shape.rank().is_static() &&
        axis != AXIS_NOT_SET_VALUE)
    {
        std::vector<Dimension> result_dims(static_cast<size_t>(params_shape.rank()) +
                                           static_cast<size_t>(indices_shape.rank()) - 1);
        size_t i = 0;
        for (; i < static_cast<size_t>(axis); i++)
        {
            result_dims[i] = params_shape[i];
        }
        for (size_t j = 0; j < static_cast<size_t>(indices_shape.rank()); i++, j++)
        {
            result_dims[i] = indices_shape[j];
        }
        for (size_t j = static_cast<size_t>(axis) + 1; j < static_cast<size_t>(params_shape.rank());
             i++, j++)
        {
            result_dims[i] = params_shape[j];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

size_t op::v1::Gather::get_axis() const
{
    int64_t axis = AXIS_NOT_SET_VALUE;
    auto axes_input_node = input_value(AXIS).get_node_shared_ptr();
    if (auto const_op = as_type_ptr<op::Constant>(axes_input_node))
    {
        axis = const_op->get_vector<int64_t>()[0];
    }
    if (axis < 0)
    {
        const auto& input_rank = get_input_partial_shape(PARAMS).rank();
        if (input_rank.is_static())
        {
            axis += static_cast<size_t>(input_rank);
        }
    }
    return static_cast<size_t>(axis);
}

void op::v1::Gather::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                       const NodeVector& /* deltas */)
{
    throw ngraph_error("Not yet implemented");
}

shared_ptr<Node> op::v1::Gather::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::Gather>(new_args.at(PARAMS), new_args.at(INDICES), new_args.at(AXIS));
}
