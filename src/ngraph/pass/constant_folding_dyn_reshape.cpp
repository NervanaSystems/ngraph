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

#include "constant_folding.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_dyn_reshape(shared_ptr<op::Constant> constant_data,
                                                   shared_ptr<op::v1::Reshape> dyn_reshape)
{
    auto out_shape = dyn_reshape->get_shape();

    AxisVector input_order(constant_data->get_shape().size());
    std::iota(input_order.begin(), input_order.end(), 0);

    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::reshape<T>(constant_data->get_data_ptr<T>(),
                                   out_vec.data(),
                                   constant_data->get_shape(),
                                   input_order,
                                   out_shape);

    return make_shared<op::Constant>(dyn_reshape->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_dyn_reshape()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto constant_shape_label =
        make_shared<pattern::op::Label>(element::i64, Shape{1}, pattern::has_class<op::Constant>());
    auto dyn_reshape = make_shared<op::v1::Reshape>(constant_data_label, constant_shape_label);

    // Note: No need to capture or consider constant_shape_label, because
    // shape propagation will have transferred the info to dyn_reshape's
    // output.
    auto constant_dyn_reshape_callback = [constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_dyn_reshape_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_data_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto dyn_reshape_match = static_pointer_cast<op::v1::Reshape>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(dyn_reshape_match));

        std::shared_ptr<Node> replacement;
        auto type = dyn_reshape_match->get_element_type();
        switch (type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_dyn_reshape_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false,
                         "Encountered 'dynamic' element type in constant_dyn_reshape_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_dyn_reshape_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_dyn_reshape<char>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::bf16:
            replacement =
                fold_constant_dyn_reshape<bfloat16>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::f16:
            replacement =
                fold_constant_dyn_reshape<float16>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_dyn_reshape<float>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_dyn_reshape<double>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_dyn_reshape<int8_t>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::i16:
            replacement =
                fold_constant_dyn_reshape<int16_t>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::i32:
            replacement =
                fold_constant_dyn_reshape<int32_t>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::i64:
            replacement =
                fold_constant_dyn_reshape<int64_t>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::u8:
            replacement =
                fold_constant_dyn_reshape<uint8_t>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::u16:
            replacement =
                fold_constant_dyn_reshape<uint16_t>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::u32:
            replacement =
                fold_constant_dyn_reshape<uint32_t>(constant_data_match, dyn_reshape_match);
            break;
        case element::Type_t::u64:
            replacement =
                fold_constant_dyn_reshape<uint64_t>(constant_data_match, dyn_reshape_match);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto dyn_reshape_matcher =
        make_shared<pattern::Matcher>(dyn_reshape, "ConstantFolding.ConstantDynReshape");
    this->add_matcher(
        dyn_reshape_matcher, constant_dyn_reshape_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
