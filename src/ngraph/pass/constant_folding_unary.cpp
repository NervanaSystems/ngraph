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
#include "ngraph/op/abs.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/any.hpp"
#include "ngraph/runtime/reference/ceiling.hpp"
#include "ngraph/runtime/reference/floor.hpp"
#include "ngraph/runtime/reference/negate.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/sign.hpp"
#include "ngraph/runtime/reference/sqrt.hpp"

using namespace std;
using namespace ngraph;

bool is_supported_unary_op(std::shared_ptr<Node> n)
{
    return n->is_type<op::Abs>() || n->is_type<op::Ceiling>() || n->is_type<op::Floor>() ||
           n->is_type<op::Negative>() || n->is_type<op::Not>() || n->is_type<op::Relu>() ||
           n->is_type<op::Sign>() || n->is_type<op::Sqrt>();
}

template <class T>
shared_ptr<op::Constant> fold_constant_unary(shared_ptr<op::Constant> constant,
                                             shared_ptr<Node> unary,
                                             NodeExecutorTy func)
{
    // check sqrt arg
    if (unary->is_type<op::Sqrt>())
    {
        std::vector<T> values{constant->get_vector<T>()};
        if (std::any_of(values.begin(), values.end(), [](T i) { return i < T(0); }))
        {
            throw ngraph_error("Square root of negative value");
        }
    }

    auto out_shape = unary->get_shape();
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
        if (unary->is_type<op::Abs>())
        {
            runtime::reference::abs<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (unary->is_type<op::Ceiling>())
        {
            runtime::reference::ceiling<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (unary->is_type<op::Floor>())
        {
            runtime::reference::floor<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (unary->is_type<op::Negative>())
        {
            runtime::reference::negate<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (unary->is_type<op::Not>())
        {
            runtime::reference::logical_not<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (unary->is_type<op::Relu>())
        {
            runtime::reference::relu<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (unary->is_type<op::Sign>())
        {
            runtime::reference::sign<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (unary->is_type<op::Sqrt>())
        {
            runtime::reference::sqrt<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else
        {
            NGRAPH_CHECK(false, "must be consistent with is_supported_unary_op");
        }
    }

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_unary()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto is_ue = [](std::shared_ptr<Node> n) {
        return n->is_unary_elementwise_arithmetic() || pattern::has_class<op::Not>()(n);
    };
    auto ue = std::make_shared<pattern::op::Any>(constant_label, is_ue, NodeVector{constant_label});

    auto constant_unary_callback = [&, constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_unary_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = as_type_ptr<op::Constant>(pattern_map[constant_label]);
        auto unary_match = m.get_match_root();

        if (!is_supported_unary_op(unary_match))
        {
            return false;
        }

        NGRAPH_CHECK(revalidate_and_ensure_static(unary_match));

        NodeExecutorTy func = nullptr;
        if (!m_cfmap.empty())
        {
            auto& node = *unary_match;
            auto handler = m_cfmap.find(type_index(typeid(node)));
            NGRAPH_CHECK(handler != m_cfmap.end(),
                         "constant folding map should have an entry for ",
                         unary_match->get_name());
            func = handler->second(unary_match.get());
        }

        std::shared_ptr<Node> replacement;
        auto type = constant_match->get_element_type();
        switch (type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_unary_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_unary_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_unary<char>(constant_match, unary_match, func);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_unary<bfloat16>(constant_match, unary_match, func);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_unary<float16>(constant_match, unary_match, func);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_unary<float>(constant_match, unary_match, func);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_unary<double>(constant_match, unary_match, func);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_unary<int8_t>(constant_match, unary_match, func);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_unary<int16_t>(constant_match, unary_match, func);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_unary<int32_t>(constant_match, unary_match, func);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_unary<int64_t>(constant_match, unary_match, func);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_unary<uint8_t>(constant_match, unary_match, func);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_unary<uint16_t>(constant_match, unary_match, func);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_unary<uint32_t>(constant_match, unary_match, func);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_unary<uint64_t>(constant_match, unary_match, func);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(ue, "ConstantFolding.ConstantUnary");
    this->add_matcher(reshape_matcher, constant_unary_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
