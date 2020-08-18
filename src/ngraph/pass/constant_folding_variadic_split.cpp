//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "constant_folding.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/variadic_split.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

void pass::ConstantFolding::construct_constant_variadic_split()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::v0::Constant>());
    auto axis_label = make_shared<pattern::op::Label>(
        element::i64, Shape{}, pattern::has_class<op::v0::Constant>());
    auto lengths_label = make_shared<pattern::op::Label>(
        element::i64, Shape{2}, pattern::has_class<op::v0::Constant>());
    auto variadic_split_pattern =
        make_shared<op::v1::VariadicSplit>(data_label, axis_label, lengths_label);

    auto constant_variadic_split_callback = [this, data_label, axis_label, lengths_label](
                                                pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_variadic_split_callback against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        const auto data_node = static_pointer_cast<op::v0::Constant>(pattern_map[data_label]);
        const auto axis_node = static_pointer_cast<op::v0::Constant>(pattern_map[axis_label]);
        const auto lengths_node = static_pointer_cast<op::v0::Constant>(pattern_map[lengths_label]);
        const auto variadic_split = m.get_match_root_as<op::v1::VariadicSplit>();
        NGRAPH_CHECK(variadic_split,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `op::v1::VariadicSplit`");

        const auto axis_val = axis_node->cast_vector<int64_t>()[0];
        const auto norm_axis_val = ngraph::normalize_axis(
            variadic_split.get(), axis_val, data_node->get_output_partial_shape(0).rank());
        auto split_lengths = lengths_node->cast_vector<int64_t>();

        // Adjust split lengths in case of negatives
        size_t sum_of_splits = 0;
        int64_t negative_one = -1;
        for (size_t i = 0; i < split_lengths.size(); i++)
        {
            if (split_lengths[i] == -1)
            {
                negative_one = i;
            }
            else
            {
                sum_of_splits += split_lengths[i];
            }
        }

        if (negative_one > 0)
        {
            split_lengths[negative_one] =
                static_cast<size_t>(data_node->get_output_shape(0)[norm_axis_val]) - sum_of_splits;
        }

        const auto slices = builder::split(
            data_node, vector<size_t>(split_lengths.begin(), split_lengths.end()), norm_axis_val);

        for (size_t i = 0; i < variadic_split->get_output_size(); i++)
        {
            for (auto& input : variadic_split->output(i).get_target_inputs())
            {
                input.replace_source_output(slices[i]);
            }
        }
        variadic_split->outputs().clear();
        construct_constant_slice();

        return true;
    };
    auto variadic_split_matcher = make_shared<pattern::Matcher>(
        variadic_split_pattern, "ConstantFolding.ConstantVariadicSplit");
    this->add_matcher(variadic_split_matcher,
                      constant_variadic_split_callback,
                      PassProperty::CHANGE_DYNAMIC_STATE);
}
