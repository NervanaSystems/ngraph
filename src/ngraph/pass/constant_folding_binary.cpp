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
#include "ngraph/op/add.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/and.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/equal.hpp"
#include "ngraph/runtime/reference/greater.hpp"
#include "ngraph/runtime/reference/greater_eq.hpp"
#include "ngraph/runtime/reference/less.hpp"
#include "ngraph/runtime/reference/less_eq.hpp"
#include "ngraph/runtime/reference/maximum.hpp"
#include "ngraph/runtime/reference/minimum.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/not_equal.hpp"
#include "ngraph/runtime/reference/or.hpp"
#include "ngraph/runtime/reference/subtract.hpp"
#include "ngraph/runtime/reference/xor.hpp"

using namespace std;
using namespace ngraph;

template <class Tin, class Tout>
shared_ptr<op::Constant> fold_constant_binary(shared_ptr<op::Constant> a,
                                              shared_ptr<op::Constant> b,
                                              shared_ptr<Node> binary,
                                              NodeExecutorTy func)
{
    auto out_shape = binary->get_shape();

    // NOTE: We will skip the executor if the shapes do not match, because that means
    // auto-broadcast is in use, and the CPU functors don't yet support that.
    if (func != nullptr && a->get_shape() == b->get_shape())
    {
        vector<Tout> out_vec(shape_size(out_shape));
        vector<void*> inputs;
        inputs.push_back(const_cast<void*>(a->get_data_ptr()));
        inputs.push_back(const_cast<void*>(b->get_data_ptr()));
        vector<void*> outputs;
        outputs.push_back(out_vec.data());

        func(inputs, outputs);
        return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
    }
    else
    {
        if (auto add_node = std::dynamic_pointer_cast<op::Add>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            vector<Tin> out_vec(shape_size(out_shape));
            runtime::reference::add<Tin>(a->get_data_ptr<Tin>(),
                                         b->get_data_ptr<Tin>(),
                                         out_vec.data(),
                                         a->get_shape(),
                                         b->get_shape(),
                                         add_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto and_node = std::dynamic_pointer_cast<op::And>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            vector<Tin> out_vec(shape_size(out_shape));
            runtime::reference::logical_and<Tin>(a->get_data_ptr<Tin>(),
                                                 b->get_data_ptr<Tin>(),
                                                 out_vec.data(),
                                                 a->get_shape(),
                                                 b->get_shape(),
                                                 and_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto divide_node = std::dynamic_pointer_cast<op::Divide>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            vector<Tin> out_vec(shape_size(out_shape));
            shared_ptr<op::Divide> divop = std::dynamic_pointer_cast<op::Divide>(binary);
            bool pythondiv = divop->is_pythondiv();
            runtime::reference::divide<Tin>(a->get_data_ptr<Tin>(),
                                            b->get_data_ptr<Tin>(),
                                            out_vec.data(),
                                            a->get_shape(),
                                            b->get_shape(),
                                            divide_node->get_autob(),
                                            pythondiv);
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto equal_node = std::dynamic_pointer_cast<op::Equal>(binary))
        {
            NGRAPH_CHECK(element::from<Tout>() == element::boolean, "Output type is not boolean");
            vector<char> out_vec(shape_size(out_shape));
            runtime::reference::equal<Tin>(a->get_data_ptr<Tin>(),
                                           b->get_data_ptr<Tin>(),
                                           out_vec.data(),
                                           a->get_shape(),
                                           b->get_shape(),
                                           equal_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto greater_node = std::dynamic_pointer_cast<op::Greater>(binary))
        {
            NGRAPH_CHECK(element::from<Tout>() == element::boolean, "Output type is not boolean");
            vector<char> out_vec(shape_size(out_shape));
            runtime::reference::greater<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             out_vec.data(),
                                             a->get_shape(),
                                             b->get_shape(),
                                             greater_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto greater_eq_node = std::dynamic_pointer_cast<op::GreaterEq>(binary))
        {
            NGRAPH_CHECK(element::from<Tout>() == element::boolean, "Output type is not boolean");
            vector<char> out_vec(shape_size(out_shape));
            runtime::reference::greater_eq<Tin>(a->get_data_ptr<Tin>(),
                                                b->get_data_ptr<Tin>(),
                                                out_vec.data(),
                                                a->get_shape(),
                                                b->get_shape(),
                                                greater_eq_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto less_node = std::dynamic_pointer_cast<op::Less>(binary))
        {
            NGRAPH_CHECK(element::from<Tout>() == element::boolean, "Output type is not boolean");
            vector<char> out_vec(shape_size(out_shape));
            runtime::reference::less<Tin>(a->get_data_ptr<Tin>(),
                                          b->get_data_ptr<Tin>(),
                                          out_vec.data(),
                                          a->get_shape(),
                                          b->get_shape(),
                                          less_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto less_eq_node = std::dynamic_pointer_cast<op::LessEq>(binary))
        {
            NGRAPH_CHECK(element::from<Tout>() == element::boolean, "Output type is not boolean");
            vector<char> out_vec(shape_size(out_shape));
            runtime::reference::less_eq<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             out_vec.data(),
                                             a->get_shape(),
                                             b->get_shape(),
                                             less_eq_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto maximum_node = std::dynamic_pointer_cast<op::Maximum>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            vector<Tin> out_vec(shape_size(out_shape));
            runtime::reference::maximum<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             out_vec.data(),
                                             a->get_shape(),
                                             b->get_shape(),
                                             maximum_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto minimum_node = std::dynamic_pointer_cast<op::Minimum>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            vector<Tin> out_vec(shape_size(out_shape));
            runtime::reference::minimum<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             out_vec.data(),
                                             a->get_shape(),
                                             b->get_shape(),
                                             minimum_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto multiply_node = std::dynamic_pointer_cast<op::Multiply>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            vector<Tin> out_vec(shape_size(out_shape));
            runtime::reference::multiply<Tin>(a->get_data_ptr<Tin>(),
                                              b->get_data_ptr<Tin>(),
                                              out_vec.data(),
                                              a->get_shape(),
                                              b->get_shape(),
                                              multiply_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto not_equal_node = std::dynamic_pointer_cast<op::NotEqual>(binary))
        {
            NGRAPH_CHECK(element::from<Tout>() == element::boolean, "Output type is not boolean");
            vector<char> out_vec(shape_size(out_shape));
            runtime::reference::not_equal<Tin>(a->get_data_ptr<Tin>(),
                                               b->get_data_ptr<Tin>(),
                                               out_vec.data(),
                                               a->get_shape(),
                                               b->get_shape(),
                                               not_equal_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto or_node = std::dynamic_pointer_cast<op::Or>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            vector<Tin> out_vec(shape_size(out_shape));
            runtime::reference::logical_or<Tin>(a->get_data_ptr<Tin>(),
                                                b->get_data_ptr<Tin>(),
                                                out_vec.data(),
                                                a->get_shape(),
                                                b->get_shape(),
                                                or_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto subtract_node = std::dynamic_pointer_cast<op::Subtract>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            vector<Tin> out_vec(shape_size(out_shape));
            runtime::reference::subtract<Tin>(a->get_data_ptr<Tin>(),
                                              b->get_data_ptr<Tin>(),
                                              out_vec.data(),
                                              a->get_shape(),
                                              b->get_shape(),
                                              subtract_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else if (auto xor_node = std::dynamic_pointer_cast<op::Xor>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            vector<Tin> out_vec(shape_size(out_shape));
            runtime::reference::logical_xor<Tin>(a->get_data_ptr<Tin>(),
                                                 b->get_data_ptr<Tin>(),
                                                 out_vec.data(),
                                                 a->get_shape(),
                                                 b->get_shape(),
                                                 xor_node->get_autob());
            return make_shared<op::Constant>(binary->get_element_type(), out_shape, out_vec);
        }
        else
        {
            NGRAPH_CHECK(false,
                         "fold_constant_binary must be consistent with is_supported_binary_op");
        }
    }
}

template <class Tin>
shared_ptr<op::Constant> fold_constant_binary_helper(const element::Type& et_out,
                                                     shared_ptr<op::Constant> a,
                                                     shared_ptr<op::Constant> b,
                                                     shared_ptr<Node> binary,
                                                     NodeExecutorTy func)
{
    switch (et_out)
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_binary_callback");
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_binary_callback");
    case element::Type_t::boolean: return fold_constant_binary<Tin, char>(a, b, binary, func);
    case element::Type_t::bf16: return fold_constant_binary<Tin, bfloat16>(a, b, binary, func);
    case element::Type_t::f16: return fold_constant_binary<Tin, float16>(a, b, binary, func);
    case element::Type_t::f32: return fold_constant_binary<Tin, float>(a, b, binary, func);
    case element::Type_t::f64: return fold_constant_binary<Tin, double>(a, b, binary, func);
    case element::Type_t::i8: return fold_constant_binary<Tin, int8_t>(a, b, binary, func);
    case element::Type_t::i16: return fold_constant_binary<Tin, int16_t>(a, b, binary, func);
    case element::Type_t::i32: return fold_constant_binary<Tin, int32_t>(a, b, binary, func);
    case element::Type_t::i64: return fold_constant_binary<Tin, int64_t>(a, b, binary, func);
    case element::Type_t::u8: return fold_constant_binary<Tin, uint8_t>(a, b, binary, func);
    case element::Type_t::u16: return fold_constant_binary<Tin, uint16_t>(a, b, binary, func);
    case element::Type_t::u32: return fold_constant_binary<Tin, uint32_t>(a, b, binary, func);
    case element::Type_t::u64: return fold_constant_binary<Tin, uint64_t>(a, b, binary, func);
    }

    NGRAPH_UNREACHABLE("Unreachable switch case");
}
bool is_supported_binary_op(std::shared_ptr<Node> n)
{
    return (std::dynamic_pointer_cast<op::Add>(n) || std::dynamic_pointer_cast<op::And>(n) ||
            std::dynamic_pointer_cast<op::Divide>(n) || std::dynamic_pointer_cast<op::Equal>(n) ||
            std::dynamic_pointer_cast<op::Greater>(n) ||
            std::dynamic_pointer_cast<op::GreaterEq>(n) || std::dynamic_pointer_cast<op::Less>(n) ||
            std::dynamic_pointer_cast<op::LessEq>(n) || std::dynamic_pointer_cast<op::Maximum>(n) ||
            std::dynamic_pointer_cast<op::Minimum>(n) ||
            std::dynamic_pointer_cast<op::Multiply>(n) ||
            std::dynamic_pointer_cast<op::NotEqual>(n) || std::dynamic_pointer_cast<op::Or>(n) ||
            std::dynamic_pointer_cast<op::Subtract>(n) || std::dynamic_pointer_cast<op::Xor>(n));
}

void pass::ConstantFolding::construct_constant_binary()
{
    auto a = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto b = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto is_be = [](std::shared_ptr<Node> n) {
        return (pattern::has_class<op::util::BinaryElementwiseArithmetic>()(n) ||
                pattern::has_class<op::util::BinaryElementwiseComparison>()(n) ||
                pattern::has_class<op::util::BinaryElementwiseLogical>()(n));
    };
    auto be = std::make_shared<pattern::op::Any>(a, is_be, NodeVector{a, b});

    auto constant_binary_callback = [&, a, b](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_binary_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto a_match = dynamic_pointer_cast<op::Constant>(pattern_map[a]);
        auto b_match = dynamic_pointer_cast<op::Constant>(pattern_map[b]);
        auto binary_match = m.get_match_root();

        if (!is_supported_binary_op(binary_match))
        {
            return false;
        }

        NGRAPH_CHECK(revalidate_and_ensure_static(binary_match));

        NodeExecutorTy func = nullptr;
        if (!m_cfmap.empty())
        {
            auto& node = *binary_match;
            auto handler = m_cfmap.find(type_index(typeid(node)));
            NGRAPH_CHECK(handler != m_cfmap.end(),
                         "constant folding map should have an entry for ",
                         binary_match->get_name());
            func = handler->second(binary_match.get());
        }

        std::shared_ptr<Node> replacement;
        auto in_type = a_match->get_output_element_type(0);
        auto out_type = binary_match->get_output_element_type(0);
        switch (in_type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_binary_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_binary_callback");
            break;
        case element::Type_t::boolean:
            replacement =
                fold_constant_binary_helper<char>(out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_binary_helper<bfloat16>(
                out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_binary_helper<float16>(
                out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::f32:
            replacement =
                fold_constant_binary_helper<float>(out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::f64:
            replacement =
                fold_constant_binary_helper<double>(out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::i8:
            replacement =
                fold_constant_binary_helper<int8_t>(out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_binary_helper<int16_t>(
                out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_binary_helper<int32_t>(
                out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_binary_helper<int64_t>(
                out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_binary_helper<uint8_t>(
                out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_binary_helper<uint16_t>(
                out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_binary_helper<uint32_t>(
                out_type, a_match, b_match, binary_match, func);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_binary_helper<uint64_t>(
                out_type, a_match, b_match, binary_match, func);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(be, "ConstantFolding.ConstantBinary");
    this->add_matcher(
        reshape_matcher, constant_binary_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
