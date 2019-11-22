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
#include "ngraph/op/fused/unsqueeze.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_unsqueeze(shared_ptr<op::Constant> constant,
                                                 shared_ptr<op::Unsqueeze> unsqueeze)
{
    auto out_shape = unsqueeze->get_shape();
    vector<T> out_vec(shape_size(out_shape));
    out_vec = constant->get_vector<T>();
    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_unsqueeze()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    Shape axes_shape{1};
    vector<int64_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto unsqueeze = make_shared<op::Unsqueeze>(constant_data_label, constant_axes);

    auto constant_unsqueeze_callback = [&, constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_unsqueeze_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto unsqueeze_match = static_pointer_cast<op::Unsqueeze>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(unsqueeze_match));

        std::shared_ptr<Node> replacement;
        auto type = constant_match->get_element_type();
        switch (type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_unsqueeze_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false,
                         "Encountered 'dynamic' element type in constant_unsqueeze_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_unsqueeze_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_unsqueeze<char>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_unsqueeze<bfloat16>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_unsqueeze<float16>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_unsqueeze<float>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_unsqueeze<double>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_unsqueeze<int8_t>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_unsqueeze<int16_t>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_unsqueeze<int32_t>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_unsqueeze<int64_t>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_unsqueeze<uint8_t>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_unsqueeze<uint16_t>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_unsqueeze<uint32_t>(constant_match, unsqueeze_match);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_unsqueeze<uint64_t>(constant_match, unsqueeze_match);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto unsqueeze_matcher =
        make_shared<pattern::Matcher>(unsqueeze, "ConstantFolding.ConstantUnsqueeze");
    this->add_matcher(
        unsqueeze_matcher, constant_unsqueeze_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
