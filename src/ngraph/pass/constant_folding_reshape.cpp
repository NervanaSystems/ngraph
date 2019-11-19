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
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/reference/reshape.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_reshape(shared_ptr<op::Constant> constant,
                                               shared_ptr<op::Reshape> reshape,
                                               NodeExecutorTy func)
{
    auto out_shape = reshape->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    if (func != nullptr)
    {
        vector<void*> inputs;
        inputs.push_back(const_cast<void*>(constant->get_data_ptr()));
        vector<void*> outputs;
        outputs.push_back(out_vec.data());

        func(inputs, outputs);
    }
    else
    {
        runtime::reference::reshape<T>(constant->get_data_ptr<T>(),
                                       out_vec.data(),
                                       constant->get_shape(),
                                       reshape->get_input_order(),
                                       out_shape);
    }

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_reshape()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto reshape = make_shared<op::Reshape>(constant_label, AxisVector{0, 1}, Shape{2, 4, 1});

    auto constant_reshape_callback = [&, constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_reshape_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto reshape_match = static_pointer_cast<op::Reshape>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(reshape_match));

        NodeExecutorTy func = nullptr;
        if (!m_cfmap.empty())
        {
            auto handler = m_cfmap.find(type_index(typeid(ngraph::op::Reshape)));
            NGRAPH_CHECK(handler != m_cfmap.end(),
                         "constant folding map should have reshape entry");
            func = handler->second(reshape_match.get());
        }

        std::shared_ptr<Node> replacement;
        auto type = constant_match->get_element_type();
        switch (type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_reshape_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_reshape_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_reshape_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_reshape<char>(constant_match, reshape_match, func);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_reshape<bfloat16>(constant_match, reshape_match, func);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_reshape<float16>(constant_match, reshape_match, func);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_reshape<float>(constant_match, reshape_match, func);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_reshape<double>(constant_match, reshape_match, func);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_reshape<int8_t>(constant_match, reshape_match, func);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_reshape<int16_t>(constant_match, reshape_match, func);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_reshape<int32_t>(constant_match, reshape_match, func);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_reshape<int64_t>(constant_match, reshape_match, func);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_reshape<uint8_t>(constant_match, reshape_match, func);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_reshape<uint16_t>(constant_match, reshape_match, func);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_reshape<uint32_t>(constant_match, reshape_match, func);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_reshape<uint64_t>(constant_match, reshape_match, func);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto reshape_matcher =
        make_shared<pattern::Matcher>(reshape, "ConstantFolding.ConstantReshape");
    this->add_matcher(
        reshape_matcher, constant_reshape_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
