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

#include "constant_folding.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/split.hpp"

using namespace std;
using namespace ngraph;

static int64_t axis_value(std::shared_ptr<op::Constant> axis_const)
{
    int64_t axis_value{0};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (static_cast<element::Type_t>(axis_const->get_element_type()))
    {
    case element::Type_t::i8: axis_value = axis_const->get_vector<int8_t>().at(0); break;
    case element::Type_t::i16: axis_value = axis_const->get_vector<int16_t>().at(0); break;
    case element::Type_t::i32: axis_value = axis_const->get_vector<int32_t>().at(0); break;
    case element::Type_t::i64: axis_value = axis_const->get_vector<int64_t>().at(0); break;
    default: break;
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    return axis_value;
}

void pass::ConstantFolding::construct_constant_split_v1()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto axis_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto split_v1 = make_shared<op::v1::Split>(data_label, axis_label, 0);

    auto constant_split_v1_callback = [this, data_label, axis_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_split_v1_callback against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        const auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        const auto axis_node = static_pointer_cast<op::Constant>(pattern_map[axis_label]);

        const auto split = static_pointer_cast<op::v1::Split>(m.get_match_root());

        const auto axis_val = axis_value(axis_node);
        const auto slices = builder::split(data_node, split->get_num_splits(), axis_val);

        for (size_t i = 0; i < split->get_output_size(); i++)
        {
            for (auto& input : split->output(i).get_target_inputs())
            {
                input.replace_source_output((slices[i]->output(0)));
            }
        }
        split->outputs().clear();
        construct_constant_slice();
        return true;
    };
    auto split_v1_matcher =
        make_shared<pattern::Matcher>(split_v1, "ConstantFolding.ConstantSplit_v1");
    this->add_matcher(
        split_v1_matcher, constant_split_v1_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
