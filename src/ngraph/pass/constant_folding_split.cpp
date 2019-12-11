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
#include "ngraph/op/fused/split.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

template <class T>
std::vector<shared_ptr<op::Constant>> fold_constant_split(shared_ptr<op::Constant> constant,
                                                          shared_ptr<op::v1::Split> split)
{
    std::vector<shared_ptr<op::Constant>> split_vec;
    auto num_splits = split->get_num_splits();
    auto in_shape = constant->get_shape();

    int64_t axis{0};
    const auto axis_input = as_type_ptr<op::Constant>(split->input_value(1).get_node_shared_ptr());

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (static_cast<element::Type_t>(axis_input->get_element_type()))
    {
    case element::Type_t::i8: axis = axis_input->get_vector<int8_t>().at(0); break;
    case element::Type_t::i32: axis = axis_input->get_vector<int32_t>().at(0); break;
    case element::Type_t::i64: axis = axis_input->get_vector<int64_t>().at(0); break;
    default: break;
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    // Adjust split axis in case of negatives
    axis = ngraph::normalize_axis(split.get(), axis, in_shape.size());

    auto out_shape = constant->get_shape();
    auto length = in_shape[axis] / num_splits;
    out_shape[axis] = length;
    std::vector<size_t> upper_bounds{out_shape};
    std::vector<size_t> lower_bounds(upper_bounds.size());
    std::vector<size_t> strides(upper_bounds.size(), 1);
    for (int i = 0; i < num_splits; i++)
    {
        lower_bounds[axis] = i * length;
        upper_bounds[axis] = (i + 1) * length;
        vector<T> out_vec(shape_size(out_shape));
        runtime::reference::slice<T>(constant->get_data_ptr<T>(),
                                     out_vec.data(),
                                     in_shape,
                                     lower_bounds,
                                     upper_bounds,
                                     strides,
                                     out_shape);

        split_vec.push_back(
            make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec));
    }
    return split_vec;
}

void pass::ConstantFolding::construct_constant_split()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto axis_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto split_op = make_shared<op::v1::Split>(data_label, axis_label, 3);

    auto constant_split_callback = [data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_split_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        auto split = static_pointer_cast<op::v1::Split>(m.get_match_root());

        std::vector<std::shared_ptr<op::Constant>> replacement_vec;

        switch (split->get_output_element_type(0))
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_split");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_split");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_split");
            break;
        case element::Type_t::boolean:
            replacement_vec = fold_constant_split<char>(data_node, split);
            break;
        case element::Type_t::bf16:
            replacement_vec = fold_constant_split<bfloat16>(data_node, split);
            break;
        case element::Type_t::f16:
            replacement_vec = fold_constant_split<float16>(data_node, split);
            break;
        case element::Type_t::f32:
            replacement_vec = fold_constant_split<float>(data_node, split);
            break;
        case element::Type_t::f64:
            replacement_vec = fold_constant_split<double>(data_node, split);
            break;
        case element::Type_t::i8:
            replacement_vec = fold_constant_split<int8_t>(data_node, split);
            break;
        case element::Type_t::i16:
            replacement_vec = fold_constant_split<int16_t>(data_node, split);
            break;
        case element::Type_t::i32:
            replacement_vec = fold_constant_split<int32_t>(data_node, split);
            break;
        case element::Type_t::i64:
            replacement_vec = fold_constant_split<int64_t>(data_node, split);
            break;
        case element::Type_t::u8:
            replacement_vec = fold_constant_split<uint8_t>(data_node, split);
            break;
        case element::Type_t::u16:
            replacement_vec = fold_constant_split<uint16_t>(data_node, split);
            break;
        case element::Type_t::u32:
            replacement_vec = fold_constant_split<uint32_t>(data_node, split);
            break;
        case element::Type_t::u64:
            replacement_vec = fold_constant_split<uint64_t>(data_node, split);
            break;
        }

        auto i = 0;
        for (auto output : split->outputs())
        {
            for (auto& input : output.get_target_inputs())
            {
                auto user = input.get_node()->shared_from_this();
                if (as_type_ptr<op::GetOutputElement>(user))
                {
                    replace_node(user, replacement_vec[i]);
                }
                else
                {
                    input.replace_source_output(replacement_vec[i]->output(0));
                }
            }
            i++;
        }
        return true;
    };

    auto split_matcher = make_shared<pattern::Matcher>(split_op, "ConstantFolding.ConstantSplit");
    this->add_matcher(split_matcher, constant_split_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
