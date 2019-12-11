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
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/variadic_split.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

template <class T>
std::vector<shared_ptr<op::Constant>>
    fold_constant_variadic_split(shared_ptr<op::Constant> constant,
                                 shared_ptr<op::v1::VariadicSplit> variadic_split)
{
    std::vector<shared_ptr<op::Constant>> split_vec;
    auto in_shape = constant->get_shape();
    const auto axis_input =
        as_type_ptr<op::Constant>(variadic_split->input_value(1).get_node_shared_ptr());
    int64_t axis = as_type_ptr<op::Constant>(axis_input)->get_vector<int64_t>()[0];

    // Adjust split axis in case of negatives
    axis = ngraph::normalize_axis(variadic_split.get(), axis, in_shape.size());

    const auto lengths_input =
        as_type_ptr<op::Constant>(variadic_split->input_value(2).get_node_shared_ptr());
    auto lengths_vec = lengths_input->get_vector<int64_t>();
    auto out_shape = constant->get_shape();
    std::vector<size_t> upper_bounds{out_shape};
    upper_bounds[axis] = 0;
    std::vector<size_t> lower_bounds(upper_bounds.size());
    std::vector<size_t> strides(upper_bounds.size(), 1);

    // Adjust split lengths in case of negatives
    size_t sum_of_splits = 0;
    int64_t negative_one = -1;
    for (size_t i = 0; i < lengths_vec.size(); i++)
    {
        if (lengths_vec[i] == -1)
        {
            negative_one = i;
        }
        else
        {
            sum_of_splits += lengths_vec[i];
        }
    }

    if (negative_one > 0)
    {
        lengths_vec[negative_one] = static_cast<size_t>(in_shape[axis]) - sum_of_splits;
    }

    for (int i = 0; i < lengths_vec.size(); i++)
    {
        out_shape[axis] = lengths_vec[i];
        lower_bounds[axis] += i == 0 ? 0 : lengths_vec[i - 1];
        upper_bounds[axis] += lengths_vec[i];
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

void pass::ConstantFolding::construct_constant_variadic_split()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto axis_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto lengths_label =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto variadic_split_op =
        make_shared<op::v1::VariadicSplit>(data_label, axis_label, lengths_label);

    auto constant_variadic_split_callback = [data_label, lengths_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_variadic_split_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        auto lengths_node = static_pointer_cast<op::Constant>(pattern_map[lengths_label]);
        auto variadic_split = static_pointer_cast<op::v1::VariadicSplit>(m.get_match_root());

        std::vector<std::shared_ptr<op::Constant>> replacement_vec;

        switch (variadic_split->get_output_element_type(0))
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in fold_constant_variadic_split");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false,
                         "Encountered 'dynamic' element type in fold_constant_variadic_split");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_variadic_split");
            break;
        case element::Type_t::boolean:
            replacement_vec = fold_constant_variadic_split<char>(data_node, variadic_split);
            break;
        case element::Type_t::bf16:
            replacement_vec = fold_constant_variadic_split<bfloat16>(data_node, variadic_split);
            break;
        case element::Type_t::f16:
            replacement_vec = fold_constant_variadic_split<float16>(data_node, variadic_split);
            break;
        case element::Type_t::f32:
            replacement_vec = fold_constant_variadic_split<float>(data_node, variadic_split);
            break;
        case element::Type_t::f64:
            replacement_vec = fold_constant_variadic_split<double>(data_node, variadic_split);
            break;
        case element::Type_t::i8:
            replacement_vec = fold_constant_variadic_split<int8_t>(data_node, variadic_split);
            break;
        case element::Type_t::i16:
            replacement_vec = fold_constant_variadic_split<int16_t>(data_node, variadic_split);
            break;
        case element::Type_t::i32:
            replacement_vec = fold_constant_variadic_split<int32_t>(data_node, variadic_split);
            break;
        case element::Type_t::i64:
            replacement_vec = fold_constant_variadic_split<int64_t>(data_node, variadic_split);
            break;
        case element::Type_t::u8:
            replacement_vec = fold_constant_variadic_split<uint8_t>(data_node, variadic_split);
            break;
        case element::Type_t::u16:
            replacement_vec = fold_constant_variadic_split<uint16_t>(data_node, variadic_split);
            break;
        case element::Type_t::u32:
            replacement_vec = fold_constant_variadic_split<uint32_t>(data_node, variadic_split);
            break;
        case element::Type_t::u64:
            replacement_vec = fold_constant_variadic_split<uint64_t>(data_node, variadic_split);
            break;
        }

        auto i = 0;
        for (auto output : variadic_split->outputs())
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

    auto variadic_split_matcher =
        make_shared<pattern::Matcher>(variadic_split_op, "ConstantFolding.ConstantVariadicSplit");
    this->add_matcher(variadic_split_matcher,
                      constant_variadic_split_callback,
                      PassProperty::CHANGE_DYNAMIC_STATE);
}
