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

#include <numeric>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/variadic_split.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::VariadicSplit::type_info;

op::v1::VariadicSplit::VariadicSplit(const Output<Node>& data,
                                     const Output<Node>& axis,
                                     const Output<Node>& split_lengths)
    : Op({data, axis, split_lengths})
{
    constructor_validate_and_infer_types();
}

void ngraph::op::v1::VariadicSplit::validate_and_infer_types()
{
    set_input_is_relevant_to_value(0);
    set_input_is_relevant_to_value(1);
    set_input_is_relevant_to_value(2);

    auto split_lengths_pshape = get_input_partial_shape(2);

    if (split_lengths_pshape.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              static_cast<size_t>(split_lengths_pshape.rank()) == 1,
                              "Split lengths should be a 1-D tensor. Got ",
                              split_lengths_pshape.rank(),
                              " instead.");

        auto num_outputs = static_cast<size_t>(split_lengths_pshape[0]);
        auto data = input_value(0);
        auto axis_input = input_value(1).get_node_shared_ptr();
        auto split_lengths_input = input_value(2).get_node_shared_ptr();
        auto data_shape = data.get_partial_shape();
        auto data_type = data.get_element_type();

        set_output_size(num_outputs);
        if (data_shape.is_static() && axis_input->is_constant() &&
            split_lengths_input->is_constant())
        {
            auto data_rank = static_cast<size_t>(data_shape.rank());
            const auto axis_input = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr());
            auto axis_val = axis_input->cast_vector<int64_t>()[0];

            // Adjust split axis in case of negatives
            int64_t axis = ngraph::normalize_axis(this, axis_val, data_rank);

            auto split_lengths =
                as_type_ptr<op::Constant>(split_lengths_input)->get_vector<int64_t>();
            // Adjust split lengths in case of negatives
            size_t sum_of_splits = 0;
            int64_t negative_one = -1;
            for (size_t i = 0; i < split_lengths.size(); i++)
            {
                NODE_VALIDATION_CHECK(this,
                                      split_lengths[i] >= -1,
                                      "Invalid value ",
                                      split_lengths[i],
                                      " in split lengths input. Should be >= -1.");

                if (split_lengths[i] == -1)
                {
                    NODE_VALIDATION_CHECK(this,
                                          negative_one == -1,
                                          "Cannot infer split with multiple -1 values at ",
                                          negative_one,
                                          " and ",
                                          i);
                    negative_one = i;
                }
                else
                {
                    sum_of_splits += split_lengths[i];
                }
            }

            if (negative_one > 0)
            {
                split_lengths[negative_one] = static_cast<size_t>(data_shape[axis]) - sum_of_splits;
                sum_of_splits += split_lengths[negative_one];
            }

            NODE_VALIDATION_CHECK(this,
                                  sum_of_splits == static_cast<size_t>(data_shape[axis]),
                                  "Total length of splits: ",
                                  sum_of_splits,
                                  " must match the length of the chosen axis: ",
                                  static_cast<size_t>(data_shape[axis]));

            for (size_t output{0}; output < num_outputs; ++output)
            {
                auto tmp_shape = data_shape.to_shape();
                tmp_shape.at(axis) = split_lengths.at(output);
                set_output_type(output, data_type, tmp_shape);
            }
        }
        else
        {
            for (size_t output{0}; output < num_outputs; ++output)
            {
                set_output_type(output, data_type, PartialShape::dynamic());
            }
        }
    }
}

shared_ptr<Node> op::v1::VariadicSplit::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::VariadicSplit>(new_args.at(0), new_args.at(1), new_args.at(2));
}
