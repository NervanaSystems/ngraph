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

#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/graph_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::ReduceProd::type_info;

op::v1::ReduceProd::ReduceProd(const Output<Node>& arg,
                               const Output<Node>& reduction_axes,
                               bool keep_dims)
    : ArithmeticReduction(arg, reduction_axes)
    , m_keep_dims{keep_dims}
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceProd::get_default_value() const
{
    return ngraph::make_constant_from_string("1", get_element_type(), get_shape());
}

shared_ptr<Node> op::v1::ReduceProd::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ReduceProd>(new_args.at(0), new_args.at(1), m_keep_dims);
}

void op::v1::ReduceProd::validate_and_infer_types()
{
    if (m_keep_dims)
    {
        auto reduction_axes = get_reduction_axes();
        auto input_shape = get_input_partial_shape(0);
        auto input_rank = input_shape.rank();
        PartialShape result_shape{PartialShape::dynamic()};

        if (input_rank.is_static() && reduction_axes_constant())
        {
            std::vector<Dimension> dims;
            for (auto axis : reduction_axes)
            {
                NODE_VALIDATION_CHECK(this,
                                      axis < size_t(input_rank),
                                      "Reduction axis (",
                                      axis,
                                      ") is out of bounds ",
                                      "(argument shape: ",
                                      input_shape,
                                      ", reduction axes: ",
                                      reduction_axes,
                                      ")");
            }
            for (size_t i = 0; i < size_t(input_rank); i++)
            {
                if (reduction_axes.count(i) == 0)
                {
                    dims.push_back(input_shape[i]);
                }
                else
                {
                    dims.push_back(Dimension{1});
                }
            }
            result_shape = PartialShape(dims);
        }
        set_input_is_relevant_to_shape(1);
        set_output_type(0, get_input_element_type(0), result_shape);
    }
    else
    {
        ArithmeticReduction::validate_and_infer_types();
    }
}
