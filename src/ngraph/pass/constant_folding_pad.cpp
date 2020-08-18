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
#include "ngraph/op/pad.hpp"
#include "ngraph/runtime/reference/pad.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::v0::Constant> fold_constant_pad(shared_ptr<op::v0::Constant> constant,
                                               shared_ptr<op::v0::Pad> pad,
                                               NodeExecutorTy func)
{
    const Shape& out_shape = pad->get_output_shape(0);
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(T));
    T* data_ptr = buffer.get_ptr<T>();
    auto pad_value = std::static_pointer_cast<op::v0::Constant>(pad->get_input_node_shared_ptr(1));

    if (func != nullptr)
    {
        vector<void*> inputs;
        inputs.push_back(const_cast<void*>(constant->get_data_ptr()));
        inputs.push_back(const_cast<void*>(pad_value->get_data_ptr()));

        vector<void*> outputs;
        outputs.push_back(data_ptr);

        func(inputs, outputs);
    }
    else
    {
        runtime::reference::pad<T>(constant->get_data_ptr<T>(),
                                   pad_value->get_data_ptr<T>(),
                                   data_ptr,
                                   constant->get_output_shape(0),
                                   out_shape,
                                   pad->get_padding_below(),
                                   pad->get_padding_above(),
                                   pad->get_pad_mode());
    }

    return make_shared<op::v0::Constant>(constant->get_output_element_type(0), out_shape, data_ptr);
}

void pass::ConstantFolding::construct_constant_pad()
{
    auto is_constant = pattern::has_class<op::v0::Constant>();
    auto constant_label = make_shared<pattern::op::Label>(element::f32, Shape{6}, is_constant);

    auto pad_value_label = make_shared<pattern::op::Label>(element::f32, Shape{}, is_constant);

    CoordinateDiff padding_below{0};
    CoordinateDiff padding_above{0};
    op::PadMode pad_mode{op::PadMode::CONSTANT};

    auto pad = make_shared<op::v0::Pad>(
        constant_label, pad_value_label, padding_below, padding_above, pad_mode);

    auto constant_pad_callback = [&, constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_pad_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::v0::Constant>(pattern_map[constant_label]);
        auto pad_match = m.get_match_root_as<op::v0::Pad>();
        NGRAPH_CHECK(
            pad_match, "match root node ", *m.get_match_root(), " not of type `op::v0::Pad`");

        NGRAPH_CHECK(revalidate_and_ensure_static(pad_match));

        NodeExecutorTy func = nullptr;
        if (!m_cfmap.empty())
        {
            auto handler = m_cfmap.find(type_index(typeid(ngraph::op::v0::Pad)));
            NGRAPH_CHECK(handler != m_cfmap.end(), "constant folding map should have pad entry");
            func = handler->second(pad_match.get());
        }

        std::shared_ptr<Node> replacement;
        auto type = constant_match->get_output_element_type(0);
        switch (type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_pad_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_pad_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_pad_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_pad<char>(constant_match, pad_match, func);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_pad<bfloat16>(constant_match, pad_match, func);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_pad<float16>(constant_match, pad_match, func);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_pad<float>(constant_match, pad_match, func);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_pad<double>(constant_match, pad_match, func);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_pad<int8_t>(constant_match, pad_match, func);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_pad<int16_t>(constant_match, pad_match, func);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_pad<int32_t>(constant_match, pad_match, func);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_pad<int64_t>(constant_match, pad_match, func);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_pad<uint8_t>(constant_match, pad_match, func);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_pad<uint16_t>(constant_match, pad_match, func);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_pad<uint32_t>(constant_match, pad_match, func);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_pad<uint64_t>(constant_match, pad_match, func);
            break;
        }

        m.get_match_value().replace(replacement->output(0));
        return true;
    };

    auto pad_matcher = make_shared<pattern::Matcher>(pad, "ConstantFolding.ConstantPad");
    this->add_matcher(pad_matcher, constant_pad_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
