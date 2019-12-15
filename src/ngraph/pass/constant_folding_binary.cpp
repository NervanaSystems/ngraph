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
#include "ngraph/op/power.hpp"
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
#include "ngraph/runtime/reference/power.hpp"
#include "ngraph/runtime/reference/subtract.hpp"
#include "ngraph/runtime/reference/xor.hpp"

using namespace std;
using namespace ngraph;

static shared_ptr<op::Constant> fold_constant_binary_logical(shared_ptr<op::Constant> a,
                                                             shared_ptr<op::Constant> b,
                                                             shared_ptr<Node> binary,
                                                             NodeExecutorTy func)
{
    const Shape& out_shape = binary->get_shape();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(char));
    char* data_ptr = buffer.get_ptr<char>();

    // NOTE: We will skip the executor if the shapes do not match, because that means
    // auto-broadcast is in use, and the CPU functors don't yet support that.
    if (func != nullptr && a->get_shape() == b->get_shape())
    {
        vector<void*> inputs;
        inputs.push_back(const_cast<void*>(a->get_data_ptr()));
        inputs.push_back(const_cast<void*>(b->get_data_ptr()));
        vector<void*> outputs;
        outputs.push_back(data_ptr);

        func(inputs, outputs);
        return make_shared<op::Constant>(binary->get_output_element_type(0), out_shape, data_ptr);
    }
    else
    {
        if (auto and_v0_node = as_type_ptr<op::v0::And>(binary))
        {
            runtime::reference::logical_and<char>(a->get_data_ptr<char>(),
                                                  b->get_data_ptr<char>(),
                                                  data_ptr,
                                                  a->get_shape(),
                                                  b->get_shape(),
                                                  and_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto logical_and_node = as_type_ptr<op::v1::LogicalAnd>(binary))
        {
            runtime::reference::logical_and<char>(a->get_data_ptr<char>(),
                                                  b->get_data_ptr<char>(),
                                                  data_ptr,
                                                  a->get_shape(),
                                                  b->get_shape(),
                                                  logical_and_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto or_node = as_type_ptr<op::v0::Or>(binary))
        {
            runtime::reference::logical_or<char>(a->get_data_ptr<char>(),
                                                 b->get_data_ptr<char>(),
                                                 data_ptr,
                                                 a->get_shape(),
                                                 b->get_shape(),
                                                 or_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto logical_or_node = as_type_ptr<op::v1::LogicalOr>(binary))
        {
            runtime::reference::logical_or<char>(a->get_data_ptr<char>(),
                                                 b->get_data_ptr<char>(),
                                                 data_ptr,
                                                 a->get_shape(),
                                                 b->get_shape(),
                                                 logical_or_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto xor_node = as_type_ptr<op::v0::Xor>(binary))
        {
            runtime::reference::logical_xor<char>(a->get_data_ptr<char>(),
                                                  b->get_data_ptr<char>(),
                                                  data_ptr,
                                                  a->get_shape(),
                                                  b->get_shape(),
                                                  xor_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto logical_xor_node = as_type_ptr<op::v1::LogicalXor>(binary))
        {
            runtime::reference::logical_xor<char>(a->get_data_ptr<char>(),
                                                  b->get_data_ptr<char>(),
                                                  data_ptr,
                                                  a->get_shape(),
                                                  b->get_shape(),
                                                  logical_xor_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else
        {
            NGRAPH_CHECK(
                false,
                "fold_constant_binary_logical must be consistent with is_supported_binary_op");
        }
    }
}

template <class Tin>
shared_ptr<op::Constant> fold_constant_binary_comparison(shared_ptr<op::Constant> a,
                                                         shared_ptr<op::Constant> b,
                                                         shared_ptr<Node> binary,
                                                         NodeExecutorTy func)
{
    const Shape& out_shape = binary->get_shape();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(char));
    char* data_ptr = buffer.get_ptr<char>();

    // NOTE: We will skip the executor if the shapes do not match, because that means
    // auto-broadcast is in use, and the CPU functors don't yet support that.
    if (func != nullptr && a->get_shape() == b->get_shape())
    {
        vector<void*> inputs;
        inputs.push_back(const_cast<void*>(a->get_data_ptr()));
        inputs.push_back(const_cast<void*>(b->get_data_ptr()));
        vector<void*> outputs;
        outputs.push_back(data_ptr);

        func(inputs, outputs);
        return make_shared<op::Constant>(binary->get_output_element_type(0), out_shape, data_ptr);
    }
    else
    {
        if (auto equal_v0_node = as_type_ptr<op::v0::Equal>(binary))
        {
            runtime::reference::equal<Tin>(a->get_data_ptr<Tin>(),
                                           b->get_data_ptr<Tin>(),
                                           data_ptr,
                                           a->get_shape(),
                                           b->get_shape(),
                                           equal_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto equal_v1_node = as_type_ptr<op::v1::Equal>(binary))
        {
            runtime::reference::equal<Tin>(a->get_data_ptr<Tin>(),
                                           b->get_data_ptr<Tin>(),
                                           data_ptr,
                                           a->get_shape(),
                                           b->get_shape(),
                                           equal_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto greater_v0_node = as_type_ptr<op::v0::Greater>(binary))
        {
            runtime::reference::greater<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             data_ptr,
                                             a->get_shape(),
                                             b->get_shape(),
                                             greater_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto greater_v1_node = as_type_ptr<op::v1::Greater>(binary))
        {
            runtime::reference::greater<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             data_ptr,
                                             a->get_shape(),
                                             b->get_shape(),
                                             greater_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto greater_eq_v0_node = as_type_ptr<op::v0::GreaterEq>(binary))
        {
            runtime::reference::greater_eq<Tin>(a->get_data_ptr<Tin>(),
                                                b->get_data_ptr<Tin>(),
                                                data_ptr,
                                                a->get_shape(),
                                                b->get_shape(),
                                                greater_eq_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto greater_eq_v1_node = as_type_ptr<op::v1::GreaterEqual>(binary))
        {
            runtime::reference::greater_eq<Tin>(a->get_data_ptr<Tin>(),
                                                b->get_data_ptr<Tin>(),
                                                data_ptr,
                                                a->get_shape(),
                                                b->get_shape(),
                                                greater_eq_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto less_v0_node = as_type_ptr<op::v0::Less>(binary))
        {
            runtime::reference::less<Tin>(a->get_data_ptr<Tin>(),
                                          b->get_data_ptr<Tin>(),
                                          data_ptr,
                                          a->get_shape(),
                                          b->get_shape(),
                                          less_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto less_v1_node = as_type_ptr<op::v1::Less>(binary))
        {
            runtime::reference::less<Tin>(a->get_data_ptr<Tin>(),
                                          b->get_data_ptr<Tin>(),
                                          data_ptr,
                                          a->get_shape(),
                                          b->get_shape(),
                                          less_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto less_eq_v0_node = as_type_ptr<op::v0::LessEq>(binary))
        {
            runtime::reference::less_eq<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             data_ptr,
                                             a->get_shape(),
                                             b->get_shape(),
                                             less_eq_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto less_eq_v1_node = as_type_ptr<op::v1::LessEqual>(binary))
        {
            runtime::reference::less_eq<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             data_ptr,
                                             a->get_shape(),
                                             b->get_shape(),
                                             less_eq_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto not_equal_v0_node = as_type_ptr<op::v0::NotEqual>(binary))
        {
            runtime::reference::not_equal<Tin>(a->get_data_ptr<Tin>(),
                                               b->get_data_ptr<Tin>(),
                                               data_ptr,
                                               a->get_shape(),
                                               b->get_shape(),
                                               not_equal_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto not_equal_v1_node = as_type_ptr<op::v1::NotEqual>(binary))
        {
            runtime::reference::not_equal<Tin>(a->get_data_ptr<Tin>(),
                                               b->get_data_ptr<Tin>(),
                                               data_ptr,
                                               a->get_shape(),
                                               b->get_shape(),
                                               not_equal_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else
        {
            NGRAPH_CHECK(false,
                         "fold_constant_binary must be consistent with is_supported_binary_op");
        }
    }
}

template <class Tin, class Tout = Tin>
shared_ptr<op::Constant> fold_constant_binary_arithmetic(shared_ptr<op::Constant> a,
                                                         shared_ptr<op::Constant> b,
                                                         shared_ptr<Node> binary,
                                                         NodeExecutorTy func)
{
    const Shape& out_shape = binary->get_shape();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(Tout));
    Tout* data_ptr = buffer.get_ptr<Tout>();

    // NOTE: We will skip the executor if the shapes do not match, because that means
    // auto-broadcast is in use, and the CPU functors don't yet support that.
    if (func != nullptr && a->get_shape() == b->get_shape())
    {
        vector<void*> inputs;
        inputs.push_back(const_cast<void*>(a->get_data_ptr()));
        inputs.push_back(const_cast<void*>(b->get_data_ptr()));
        vector<void*> outputs;
        outputs.push_back(data_ptr);

        func(inputs, outputs);
        return make_shared<op::Constant>(binary->get_output_element_type(0), out_shape, data_ptr);
    }
    else
    {
        if (auto add_v0_node = as_type_ptr<op::v0::Add>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            runtime::reference::add<Tin>(a->get_data_ptr<Tin>(),
                                         b->get_data_ptr<Tin>(),
                                         data_ptr,
                                         a->get_shape(),
                                         b->get_shape(),
                                         add_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto add_v1_node = as_type_ptr<op::v1::Add>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            runtime::reference::add<Tin>(a->get_data_ptr<Tin>(),
                                         b->get_data_ptr<Tin>(),
                                         data_ptr,
                                         a->get_shape(),
                                         b->get_shape(),
                                         add_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto divide_v0_node = as_type_ptr<op::v0::Divide>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            shared_ptr<op::v0::Divide> divop = as_type_ptr<op::v0::Divide>(binary);
            bool pythondiv = divop->is_pythondiv();
            runtime::reference::divide<Tin>(a->get_data_ptr<Tin>(),
                                            b->get_data_ptr<Tin>(),
                                            data_ptr,
                                            a->get_shape(),
                                            b->get_shape(),
                                            divide_v0_node->get_autob(),
                                            pythondiv);
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto divide_v1_node = as_type_ptr<op::v1::Divide>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            shared_ptr<op::v1::Divide> divop = as_type_ptr<op::v1::Divide>(binary);
            bool pythondiv = divop->is_pythondiv();
            runtime::reference::divide<Tin>(a->get_data_ptr<Tin>(),
                                            b->get_data_ptr<Tin>(),
                                            data_ptr,
                                            a->get_shape(),
                                            b->get_shape(),
                                            divide_v1_node->get_autob(),
                                            pythondiv);
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto maximum_v0_node = as_type_ptr<op::v0::Maximum>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            runtime::reference::maximum<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             data_ptr,
                                             a->get_shape(),
                                             b->get_shape(),
                                             maximum_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto maximum_v1_node = as_type_ptr<op::v1::Maximum>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            runtime::reference::maximum<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             data_ptr,
                                             a->get_shape(),
                                             b->get_shape(),
                                             maximum_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto minimum_v0_node = as_type_ptr<op::v0::Minimum>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            runtime::reference::minimum<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             data_ptr,
                                             a->get_shape(),
                                             b->get_shape(),
                                             minimum_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto minimum_v1_node = as_type_ptr<op::v1::Minimum>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            runtime::reference::minimum<Tin>(a->get_data_ptr<Tin>(),
                                             b->get_data_ptr<Tin>(),
                                             data_ptr,
                                             a->get_shape(),
                                             b->get_shape(),
                                             minimum_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto multiply_v0_node = as_type_ptr<op::v0::Multiply>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            runtime::reference::multiply<Tin>(a->get_data_ptr<Tin>(),
                                              b->get_data_ptr<Tin>(),
                                              data_ptr,
                                              a->get_shape(),
                                              b->get_shape(),
                                              multiply_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto multiply_v1_node = as_type_ptr<op::v1::Multiply>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            runtime::reference::multiply<Tin>(a->get_data_ptr<Tin>(),
                                              b->get_data_ptr<Tin>(),
                                              data_ptr,
                                              a->get_shape(),
                                              b->get_shape(),
                                              multiply_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto power_v0_node = as_type_ptr<op::v0::Power>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            shared_ptr<op::v0::Power> powop = as_type_ptr<op::v0::Power>(binary);
            runtime::reference::power<Tin>(a->get_data_ptr<Tin>(),
                                           b->get_data_ptr<Tin>(),
                                           data_ptr,
                                           a->get_shape(),
                                           b->get_shape(),
                                           power_v0_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto power_v1_node = as_type_ptr<op::v1::Power>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            shared_ptr<op::v1::Power> powop = as_type_ptr<op::v1::Power>(binary);
            runtime::reference::power<Tin>(a->get_data_ptr<Tin>(),
                                           b->get_data_ptr<Tin>(),
                                           data_ptr,
                                           a->get_shape(),
                                           b->get_shape(),
                                           power_v1_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else if (auto subtract_node = as_type_ptr<op::Subtract>(binary))
        {
            NGRAPH_CHECK(element::from<Tin>() == element::from<Tout>(),
                         "Input/output types do not match");
            runtime::reference::subtract<Tin>(a->get_data_ptr<Tin>(),
                                              b->get_data_ptr<Tin>(),
                                              data_ptr,
                                              a->get_shape(),
                                              b->get_shape(),
                                              subtract_node->get_autob());
            return make_shared<op::Constant>(
                binary->get_output_element_type(0), out_shape, data_ptr);
        }
        else
        {
            NGRAPH_CHECK(false,
                         "fold_constant_binary must be consistent with is_supported_binary_op");
        }
    }
}

template <class Tin>
shared_ptr<op::Constant> fold_constant_binary_helper(shared_ptr<op::Constant> a,
                                                     shared_ptr<op::Constant> b,
                                                     shared_ptr<Node> binary,
                                                     NodeExecutorTy func)
{
    if (binary->is_binary_elementwise_comparison())
    {
        return fold_constant_binary_comparison<Tin>(a, b, binary, func);
    }
    else if (binary->is_binary_elementwise_arithmetic())
    {
        return fold_constant_binary_arithmetic<Tin>(a, b, binary, func);
    }
    else
    {
        NGRAPH_CHECK(
            false, "fold_constant_binary_helper only available for comparison and arithmetic ops");
    }
}

bool is_supported_binary_op(std::shared_ptr<Node> n)
{
    return (
        is_type<op::v0::Add>(n) || is_type<op::v1::Add>(n) || is_type<op::v0::Multiply>(n) ||
        is_type<op::v1::Multiply>(n) || is_type<op::v0::Divide>(n) || is_type<op::v1::Divide>(n) ||
        is_type<op::v0::Power>(n) || is_type<op::v1::Power>(n) || is_type<op::v0::Equal>(n) ||
        is_type<op::v1::Equal>(n) || is_type<op::v0::NotEqual>(n) || is_type<op::v1::NotEqual>(n) ||
        is_type<op::v0::Greater>(n) || is_type<op::v1::Greater>(n) ||
        is_type<op::v0::GreaterEq>(n) || is_type<op::v1::GreaterEqual>(n) ||
        is_type<op::v0::Less>(n) || is_type<op::v1::Less>(n) || is_type<op::v0::LessEq>(n) ||
        is_type<op::v1::LessEqual>(n) || is_type<op::v0::Maximum>(n) ||
        is_type<op::v1::Maximum>(n) || is_type<op::v0::Minimum>(n) || is_type<op::v1::Minimum>(n) ||
        is_type<op::v0::And>(n) || is_type<op::v1::LogicalAnd>(n) || is_type<op::v0::Or>(n) ||
        is_type<op::v1::LogicalOr>(n) || is_type<op::v0::Xor>(n) ||
        is_type<op::v1::LogicalXor>(n) || is_type<op::Subtract>(n));
}

void pass::ConstantFolding::construct_constant_binary()
{
    auto a = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto b = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto is_be = [](std::shared_ptr<Node> n) {
        return (n->is_binary_elementwise_arithmetic() || n->is_binary_elementwise_comparison() ||
                n->is_binary_elementwise_logical());
    };
    auto be = std::make_shared<pattern::op::Any>(a, is_be, NodeVector{a, b});

    auto constant_binary_callback = [&, a, b](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_binary_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto a_match = as_type_ptr<op::Constant>(pattern_map[a]);
        auto b_match = as_type_ptr<op::Constant>(pattern_map[b]);
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

        if (binary_match->is_binary_elementwise_logical())
        {
            replacement = fold_constant_binary_logical(a_match, b_match, binary_match, func);
        }
        else
        {
            auto in_type = a_match->get_output_element_type(0);
            auto out_type = binary_match->get_output_element_type(0);
            switch (in_type)
            {
            case element::Type_t::undefined:
                NGRAPH_CHECK(false,
                             "Encountered 'undefined' element type in constant_binary_callback");
                break;
            case element::Type_t::dynamic:
                NGRAPH_CHECK(false,
                             "Encountered 'dynamic' element type in constant_binary_callback");
                break;
            case element::Type_t::u1:
                NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_binary_callback");
                break;
            case element::Type_t::boolean:
                replacement =
                    fold_constant_binary_helper<char>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::bf16:
                replacement =
                    fold_constant_binary_helper<bfloat16>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::f16:
                replacement =
                    fold_constant_binary_helper<float16>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::f32:
                replacement =
                    fold_constant_binary_helper<float>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::f64:
                replacement =
                    fold_constant_binary_helper<double>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::i8:
                replacement =
                    fold_constant_binary_helper<int8_t>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::i16:
                replacement =
                    fold_constant_binary_helper<int16_t>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::i32:
                replacement =
                    fold_constant_binary_helper<int32_t>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::i64:
                replacement =
                    fold_constant_binary_helper<int64_t>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::u8:
                replacement =
                    fold_constant_binary_helper<uint8_t>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::u16:
                replacement =
                    fold_constant_binary_helper<uint16_t>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::u32:
                replacement =
                    fold_constant_binary_helper<uint32_t>(a_match, b_match, binary_match, func);
                break;
            case element::Type_t::u64:
                replacement =
                    fold_constant_binary_helper<uint64_t>(a_match, b_match, binary_match, func);
                break;
            }
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(be, "ConstantFolding.ConstantBinary");
    this->add_matcher(
        reshape_matcher, constant_binary_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
