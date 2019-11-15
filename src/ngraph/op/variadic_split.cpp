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

    auto split_lengths_pshape_rank = get_input_partial_shape(2).rank();

    if (split_lengths_pshape_rank.is_static())
    {
        auto num_outputs = static_cast<size_t>(split_lengths_pshape_rank);
        auto data = input_value(0);
        auto axis_input = input_value(1).get_node_shared_ptr();
        auto split_lengths_input = input_value(2).get_node_shared_ptr();
        auto data_shape = data.get_partial_shape();
        auto data_type = data.get_element_type();

        set_output_size(num_outputs);
        if (data_shape.is_static() && axis_input->is_constant() &&
            split_lengths_input->is_constant())
        {
            auto axis = as_type_ptr<op::Constant>(axis_input)->get_vector<size_t>()[0];
            auto split_lengths = as_type_ptr<op::Constant>(axis_input)->get_vector<size_t>();

            auto splits_length = std::accumulate(split_lengths.begin(), split_lengths.end(), 0UL);

            NODE_VALIDATION_CHECK(this, axis > 0, "Provided axis:", axis, " can not be negative");
            auto data_rank = static_cast<size_t>(data_shape.rank());
            NODE_VALIDATION_CHECK(this,
                                  axis < data_rank,
                                  "Provided axis:",
                                  axis,
                                  " can not be higher than input data rank: ",
                                  data_rank);

            NODE_VALIDATION_CHECK(this,
                                  splits_length == static_cast<size_t>(data_shape[axis]),
                                  "Total length of splits:",
                                  splits_length,
                                  " does not sum to length of the choosen axis: ",
                                  static_cast<size_t>(data_shape[axis]));

            for (size_t output{0}; output < num_outputs; ++output)
            {
                auto tmp_shape = data_shape.to_shape();
                tmp_shape.at(axis) = split_lengths.at(axis);
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
