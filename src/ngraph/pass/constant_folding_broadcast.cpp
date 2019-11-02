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
#include "ngraph/op/broadcast.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_broadcast(shared_ptr<op::Constant> constant,
                                                 shared_ptr<Node> broadcast,
                                                 NodeExecutorTy func)
{
    auto out_shape = broadcast->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    if (func != nullptr)
    {
        vector<void*> inputs;
        inputs.push_back(const_cast<void*>(constant->get_data_ptr()));
        vector<void*> outputs;
        outputs.push_back(out_vec.data());

        func(inputs, outputs);
    }
    else if (auto broadcast_v1 = as_type_ptr<op::v1::Broadcast>(broadcast))
    {
        auto static_bcast_axes = broadcast_v1->get_broadcast_axes();
        if (static_bcast_axes.first)
        {
            runtime::reference::broadcast<T>(constant->get_data_ptr<T>(),
                                             out_vec.data(),
                                             constant->get_shape(),
                                             out_shape,
                                             static_bcast_axes.second);
        }
        else
        {
            throw ngraph_error("Unexpected failure due to inability to obtain broadcast axes.");
        }
    }
    else if (auto broadcast_v0 = as_type_ptr<op::v0::Broadcast>(broadcast))
    {
        runtime::reference::broadcast<T>(constant->get_data_ptr<T>(),
                                         out_vec.data(),
                                         constant->get_shape(),
                                         out_shape,
                                         broadcast_v0->get_broadcast_axes());
    }
    else
    {
        throw ngraph_error("Unsupported op in broadcast constant folding.");
    }

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_broadcast()
{
    auto constant_label =
        make_shared<pattern::op::Label>(element::f32, Shape{2}, pattern::has_class<op::Constant>());

    auto broadcast_v0 = make_shared<op::v0::Broadcast>(constant_label, Shape{2, 4}, AxisSet{1});

    auto constant_shape =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto constant_axes =
        make_shared<pattern::op::Label>(element::i64, Shape{1}, pattern::has_class<op::Constant>());
    auto broadcast_v1 =
        make_shared<op::v1::Broadcast>(constant_label, constant_shape, constant_axes);

    auto constant_broadcast_callback = [&, constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_broadcast_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto broadcast_match = m.get_match_root();

        NGRAPH_CHECK(revalidate_and_ensure_static(broadcast_match));

        NodeExecutorTy func = nullptr;
        if (!m_cfmap.empty())
        {
            auto handler = m_cfmap.find(type_index(typeid(ngraph::op::Broadcast)));
            NGRAPH_CHECK(handler != m_cfmap.end(),
                         "constant folding map should have broadcast entry");
            func = handler->second(broadcast_match.get());
        }

        std::shared_ptr<Node> replacement;
        auto type = broadcast_match->get_element_type();
        switch (type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_broadcast_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false,
                         "Encountered 'dynamic' element type in constant_broadcast_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_broadcast<char>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_broadcast<bfloat16>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_broadcast<float16>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_broadcast<float>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_broadcast<double>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_broadcast<int8_t>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_broadcast<int16_t>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_broadcast<int32_t>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_broadcast<int64_t>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_broadcast<uint8_t>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_broadcast<uint16_t>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_broadcast<uint32_t>(constant_match, broadcast_match, func);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_broadcast<uint64_t>(constant_match, broadcast_match, func);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    this->add_matcher(
        make_shared<pattern::Matcher>(broadcast_v0, "ConstantFolding.ConstantBroadcastV0"),
        constant_broadcast_callback,
        PassProperty::CHANGE_DYNAMIC_STATE);

    this->add_matcher(
        make_shared<pattern::Matcher>(broadcast_v1, "ConstantFolding.ConstantBroadcastV1"),
        constant_broadcast_callback,
        PassProperty::CHANGE_DYNAMIC_STATE);
}
