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
#include "ngraph/op/fused/squeeze.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_squeeze(shared_ptr<op::Constant> constant,
                                               shared_ptr<op::Squeeze> squeeze)
{
    const Shape& out_shape = squeeze->get_shape();
    return make_shared<op::Constant>(
        constant->get_element_type(), out_shape, constant->get_data_ptr());
}

void pass::ConstantFolding::construct_constant_squeeze()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 1, 4}, pattern::has_class<op::Constant>());
    Shape axes_shape{1};
    vector<int64_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto squeeze = make_shared<op::Squeeze>(constant_data_label, constant_axes);

    auto constant_squeeze_callback = [&, constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_squeeze_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto squeeze_match = static_pointer_cast<op::Squeeze>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(squeeze_match));

        std::shared_ptr<Node> replacement;
        auto type = constant_match->get_element_type();
        switch (type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_squeeze_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_squeeze_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_squeeze_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_squeeze<char>(constant_match, squeeze_match);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_squeeze<bfloat16>(constant_match, squeeze_match);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_squeeze<float16>(constant_match, squeeze_match);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_squeeze<float>(constant_match, squeeze_match);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_squeeze<double>(constant_match, squeeze_match);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_squeeze<int8_t>(constant_match, squeeze_match);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_squeeze<int16_t>(constant_match, squeeze_match);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_squeeze<int32_t>(constant_match, squeeze_match);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_squeeze<int64_t>(constant_match, squeeze_match);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_squeeze<uint8_t>(constant_match, squeeze_match);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_squeeze<uint16_t>(constant_match, squeeze_match);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_squeeze<uint32_t>(constant_match, squeeze_match);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_squeeze<uint64_t>(constant_match, squeeze_match);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto squeeze_matcher =
        make_shared<pattern::Matcher>(squeeze, "ConstantFolding.ConstantSqueeze");
    this->add_matcher(
        squeeze_matcher, constant_squeeze_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
