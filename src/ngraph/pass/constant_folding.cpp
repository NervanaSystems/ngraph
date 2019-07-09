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

#include <stdint.h>

#include "constant_folding.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/convert.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/maximum.hpp"
#include "ngraph/runtime/reference/minimum.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/negate.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/quantize.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/sqrt.hpp"
#include "ngraph/runtime/reference/subtract.hpp"
#include "ngraph/util.hpp"

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

template <class T>
shared_ptr<op::Constant> fold_constant_pad(shared_ptr<op::Constant> constant,
                                           shared_ptr<op::Pad> pad,
                                           NodeExecutorTy func)
{
    auto out_shape = pad->get_shape();
    vector<T> out_vec(shape_size(out_shape));
    auto pad_value = std::static_pointer_cast<op::Constant>(pad->get_argument(1));

    if (func != nullptr)
    {
        vector<void*> inputs;
        inputs.push_back(const_cast<void*>(constant->get_data_ptr()));
        inputs.push_back(const_cast<void*>(pad_value->get_data_ptr()));

        vector<void*> outputs;
        outputs.push_back(out_vec.data());

        func(inputs, outputs);
    }
    else
    {
        runtime::reference::pad<T>(constant->get_data_ptr<T>(),
                                   pad_value->get_data_ptr<T>(),
                                   out_vec.data(),
                                   constant->get_shape(),
                                   out_shape,
                                   pad->get_padding_below(),
                                   pad->get_padding_above(),
                                   pad->get_pad_mode());
    }

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_pad()
{
    auto is_constant = pattern::has_class<op::Constant>();
    auto constant_label = make_shared<pattern::op::Label>(element::f32, Shape{6}, is_constant);

    auto pad_value_label = make_shared<pattern::op::Label>(element::f32, Shape{}, is_constant);

    CoordinateDiff padding_below{0};
    CoordinateDiff padding_above{0};
    op::PadMode pad_mode{op::PadMode::CONSTANT};

    auto pad = make_shared<op::Pad>(
        constant_label, pad_value_label, padding_below, padding_above, pad_mode);

    auto constant_pad_callback = [&, constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_pad_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto pad_match = static_pointer_cast<op::Pad>(m.get_match_root());

        NodeExecutorTy func = nullptr;
        if (!m_cfmap.empty())
        {
            auto handler = m_cfmap.find(type_index(typeid(ngraph::op::Pad)));
            NGRAPH_CHECK(handler != m_cfmap.end(), "constant folding map should have pad entry");
            func = handler->second(pad_match.get());
        }

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_pad<int>(constant_match, pad_match, func));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         fold_constant_pad<int8_t>(constant_match, pad_match, func));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_pad<float>(constant_match, pad_match, func));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         fold_constant_pad<double>(constant_match, pad_match, func));
            return true;
        }

        return false;
    };

    auto pad_matcher = make_shared<pattern::Matcher>(pad, "ConstantFolding.ConstantPad");
    this->add_matcher(pad_matcher, constant_pad_callback, PassProperty::REQUIRE_STATIC_SHAPE);
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

        NodeExecutorTy func = nullptr;
        if (!m_cfmap.empty())
        {
            auto handler = m_cfmap.find(type_index(typeid(ngraph::op::Reshape)));
            NGRAPH_CHECK(handler != m_cfmap.end(),
                         "constant folding map should have reshape entry");
            func = handler->second(reshape_match.get());
        }

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_reshape<int32_t>(constant_match, reshape_match, func));
            return true;
        }
        if (type == element::i64)
        {
            replace_node(m.get_match_root(),
                         fold_constant_reshape<int64_t>(constant_match, reshape_match, func));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         fold_constant_reshape<int8_t>(constant_match, reshape_match, func));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_reshape<float>(constant_match, reshape_match, func));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         fold_constant_reshape<double>(constant_match, reshape_match, func));
            return true;
        }
        else if (type == element::bf16)
        {
            replace_node(
                m.get_match_root(),
                fold_constant_reshape<ngraph::bfloat16>(constant_match, reshape_match, func));
            return true;
        }

        return false;
    };

    auto reshape_matcher =
        make_shared<pattern::Matcher>(reshape, "ConstantFolding.ConstantReshape");
    this->add_matcher(
        reshape_matcher, constant_reshape_callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

template <class T>
shared_ptr<op::Constant> fold_constant_broadcast(shared_ptr<op::Constant> constant,
                                                 shared_ptr<op::Broadcast> broadcast,
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
    else
    {
        runtime::reference::broadcast<T>(constant->get_data_ptr<T>(),
                                         out_vec.data(),
                                         constant->get_shape(),
                                         out_shape,
                                         broadcast->get_broadcast_axes());
    }

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_broadcast()
{
    auto constant_label =
        make_shared<pattern::op::Label>(element::f32, Shape{2}, pattern::has_class<op::Constant>());

    auto broadcast = make_shared<op::Broadcast>(constant_label, Shape{2, 4}, AxisSet{1});

    auto constant_broadcast_callback = [&, constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_broadcast_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto broadcast_match = static_pointer_cast<op::Broadcast>(m.get_match_root());

        NodeExecutorTy func = nullptr;
        if (!m_cfmap.empty())
        {
            auto handler = m_cfmap.find(type_index(typeid(ngraph::op::Broadcast)));
            NGRAPH_CHECK(handler != m_cfmap.end(),
                         "constant folding map should have broadcast entry");
            func = handler->second(broadcast_match.get());
        }

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_broadcast<int>(constant_match, broadcast_match, func));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         fold_constant_broadcast<int8_t>(constant_match, broadcast_match, func));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_broadcast<float>(constant_match, broadcast_match, func));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         fold_constant_broadcast<double>(constant_match, broadcast_match, func));
            return true;
        }
        else if (type == element::bf16)
        {
            replace_node(
                m.get_match_root(),
                fold_constant_broadcast<ngraph::bfloat16>(constant_match, broadcast_match, func));
            return true;
        }

        return false;
    };

    auto broadcast_matcher =
        make_shared<pattern::Matcher>(broadcast, "ConstantFolding.ConstantBroadcast");
    this->add_matcher(
        broadcast_matcher, constant_broadcast_callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

template <class T>
shared_ptr<op::Constant> fold_constant_binary(shared_ptr<op::Constant> a,
                                              shared_ptr<op::Constant> b,
                                              shared_ptr<Node> binary,
                                              NodeExecutorTy func)
{
    auto out_shape = binary->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    if (func != nullptr)
    {
        vector<void*> inputs;
        inputs.push_back(const_cast<void*>(a->get_data_ptr()));
        inputs.push_back(const_cast<void*>(b->get_data_ptr()));
        vector<void*> outputs;
        outputs.push_back(out_vec.data());

        func(inputs, outputs);
    }
    else
    {
        if (std::dynamic_pointer_cast<op::Add>(binary))
        {
            runtime::reference::add<T>(
                a->get_data_ptr<T>(), b->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Subtract>(binary))
        {
            runtime::reference::subtract<T>(
                a->get_data_ptr<T>(), b->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Multiply>(binary))
        {
            runtime::reference::multiply<T>(
                a->get_data_ptr<T>(), b->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Divide>(binary))
        {
            shared_ptr<op::Divide> divop = std::dynamic_pointer_cast<op::Divide>(binary);
            bool pythondiv = divop->is_pythondiv();
            runtime::reference::divide<T>(a->get_data_ptr<T>(),
                                          b->get_data_ptr<T>(),
                                          out_vec.data(),
                                          shape_size(out_shape),
                                          pythondiv);
        }
        else if (std::dynamic_pointer_cast<op::Minimum>(binary))
        {
            runtime::reference::minimum<T>(
                a->get_data_ptr<T>(), b->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Maximum>(binary))
        {
            runtime::reference::maximum<T>(
                a->get_data_ptr<T>(), b->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else
        {
            NGRAPH_CHECK(false,
                         "fold_constant_binary must be consistent with is_supported_binary_op");
        }
    }

    return make_shared<op::Constant>(a->get_element_type(), out_shape, out_vec);
}

bool is_supported_binary_op(std::shared_ptr<Node> n)
{
    return (std::dynamic_pointer_cast<op::Add>(n) || std::dynamic_pointer_cast<op::Subtract>(n) ||
            std::dynamic_pointer_cast<op::Multiply>(n) ||
            std::dynamic_pointer_cast<op::Divide>(n) || std::dynamic_pointer_cast<op::Maximum>(n) ||
            std::dynamic_pointer_cast<op::Minimum>(n));
}

void pass::ConstantFolding::construct_constant_binary()
{
    auto a = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto b = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto is_bea = pattern::has_class<op::util::BinaryElementwiseArithmetic>();
    auto bea = std::make_shared<pattern::op::Any>(a, is_bea, NodeVector{a, b});

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

        auto type = a_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_binary<int>(a_match, b_match, binary_match, func));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         fold_constant_binary<int8_t>(a_match, b_match, binary_match, func));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_binary<float>(a_match, b_match, binary_match, func));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         fold_constant_binary<double>(a_match, b_match, binary_match, func));
            return true;
        }

        return false;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(bea, "ConstantFolding.ConstantBinary");
    this->add_matcher(
        reshape_matcher, constant_binary_callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

bool is_supported_unary_op(std::shared_ptr<Node> n)
{
    return std::dynamic_pointer_cast<op::Abs>(n) || std::dynamic_pointer_cast<op::Negative>(n) ||
           std::dynamic_pointer_cast<op::Relu>(n) || std::dynamic_pointer_cast<op::Sqrt>(n);
}

template <class T>
shared_ptr<op::Constant> fold_constant_unary(shared_ptr<op::Constant> constant,
                                             shared_ptr<Node> unary,
                                             NodeExecutorTy func)
{
    //check sqrt arg
    if (std::dynamic_pointer_cast<op::Sqrt>(unary))
    {
        std::vector<T> values{constant->get_vector<T>()};
        if (std::any_of(values.begin(), values.end(), [](T i) { return i < 0; }))
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
        if (std::dynamic_pointer_cast<op::Abs>(unary))
        {
            runtime::reference::abs<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Negative>(unary))
        {
            runtime::reference::negate<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Relu>(unary))
        {
            runtime::reference::relu<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Sqrt>(unary))
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
    auto is_uea = pattern::has_class<op::util::UnaryElementwiseArithmetic>();
    auto uea =
        std::make_shared<pattern::op::Any>(constant_label, is_uea, NodeVector{constant_label});

    auto constant_unary_callback = [&, constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_unary_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = dynamic_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto unary_match = m.get_match_root();

        if (!is_supported_unary_op(unary_match))
        {
            return false;
        }

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

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_unary<int>(constant_match, unary_match, func));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         fold_constant_unary<int8_t>(constant_match, unary_match, func));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         fold_constant_unary<float>(constant_match, unary_match, func));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         fold_constant_unary<double>(constant_match, unary_match, func));
            return true;
        }

        return false;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(uea, "ConstantFolding.ConstantUnary");
    this->add_matcher(reshape_matcher, constant_unary_callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

template <class QUANT, class REAL>
shared_ptr<op::Constant> fold_constant_dequantize(shared_ptr<op::Constant> constant,
                                                  shared_ptr<op::Dequantize> dequant,
                                                  shared_ptr<op::Constant> scale,
                                                  shared_ptr<op::Constant> offset)
{
    auto out_shape = constant->get_shape();
    vector<REAL> out_vec(shape_size(out_shape));

    runtime::reference::dequantize<QUANT, REAL>(constant->get_vector<QUANT>().data(),
                                                scale->get_vector<REAL>().data(),
                                                offset->get_vector<QUANT>().data(),
                                                out_vec.data(),
                                                constant->get_shape(),
                                                scale->get_shape(),
                                                dequant->get_axes());

    return make_shared<op::Constant>(dequant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_dequantize()
{
    auto constant_label =
        make_shared<pattern::op::Label>(element::u8, Shape{2}, pattern::has_class<op::Constant>());
    auto dq_scale = op::Constant::create(element::f32, Shape{}, {1});
    auto dq_offset = op::Constant::create(element::u8, Shape{}, {1});
    auto dequant_op =
        make_shared<op::Dequantize>(constant_label, dq_scale, dq_offset, element::f32, AxisSet{});
    auto dequant = make_shared<pattern::op::Label>(dequant_op, nullptr, NodeVector{dequant_op});

    auto constant_dequantize_callback = [constant_label, dequant](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_dequantize_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = dynamic_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto dequant_match = pattern_map[dequant];
        auto dequantize_op = dynamic_pointer_cast<op::Dequantize>(dequant_match);
        auto args = dequant_match->get_arguments();
        auto scale = dynamic_pointer_cast<op::Constant>(args[1]);
        auto offset = dynamic_pointer_cast<op::Constant>(args[2]);

        auto type = constant_match->get_element_type();

        if (dequant_match->get_element_type() != element::f32)
        {
            return false;
        }

        if (type == element::u8)
        {
            replace_node(m.get_match_root(),
                         fold_constant_dequantize<uint8_t, float>(
                             constant_match, dequantize_op, scale, offset));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         fold_constant_dequantize<int8_t, float>(
                             constant_match, dequantize_op, scale, offset));
            return true;
        }

        return false;
    };

    auto dequantize_matcher =
        make_shared<pattern::Matcher>(dequant, "ConstantFolding.ConstantDequantize");
    this->add_matcher(
        dequantize_matcher, constant_dequantize_callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

template <class REAL, class QUANT>
shared_ptr<op::Constant> fold_constant_quantize(shared_ptr<op::Constant> constant,
                                                shared_ptr<op::Quantize> quant,
                                                shared_ptr<op::Constant> scale,
                                                shared_ptr<op::Constant> offset)
{
    auto out_shape = constant->get_shape();
    vector<QUANT> out_vec(shape_size(out_shape));

    runtime::reference::quantize<REAL, QUANT>(constant->get_vector<REAL>().data(),
                                              scale->get_vector<REAL>().data(),
                                              offset->get_vector<QUANT>().data(),
                                              out_vec.data(),
                                              constant->get_shape(),
                                              scale->get_shape(),
                                              quant->get_axes(),
                                              quant->get_round_mode());

    return make_shared<op::Constant>(quant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_quantize()
{
    auto constant_label =
        make_shared<pattern::op::Label>(element::f32, Shape{2}, pattern::has_class<op::Constant>());
    auto q_scale = op::Constant::create(element::f32, Shape{}, {1});
    auto q_offset = op::Constant::create(element::i8, Shape{}, {0});
    auto mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY;
    auto quant_op =
        make_shared<op::Quantize>(constant_label, q_scale, q_offset, element::i8, AxisSet{}, mode);
    auto quant = make_shared<pattern::op::Label>(quant_op, nullptr, NodeVector{quant_op});

    auto constant_quantize_callback = [constant_label, quant](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_quantize_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = dynamic_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto quant_match = pattern_map[quant];
        auto quantize_op = dynamic_pointer_cast<op::Quantize>(quant_match);
        auto args = quant_match->get_arguments();
        auto scale = static_pointer_cast<op::Constant>(args[1]);
        auto offset = static_pointer_cast<op::Constant>(args[2]);

        auto type = quant_match->get_element_type();

        if (constant_match->get_element_type() != element::f32)
        {
            return false;
        }

        if (type == element::u8)
        {
            replace_node(
                m.get_match_root(),
                fold_constant_quantize<float, uint8_t>(constant_match, quantize_op, scale, offset));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(
                m.get_match_root(),
                fold_constant_quantize<float, int8_t>(constant_match, quantize_op, scale, offset));
            return true;
        }

        return false;
    };

    auto quantize_matcher =
        make_shared<pattern::Matcher>(quant, "ConstantFolding.ConstantQuantize");
    this->add_matcher(
        quantize_matcher, constant_quantize_callback, PassProperty::REQUIRE_STATIC_SHAPE);
}

// Helper for mapping element::Types to runtime::reference::convert, which is templated in C++
// data types. Used by fold_constant_convert and fold_constant_convert_helper0, which respectively
// determine the appropriate C++ types for "TI" (input type) and "TO" (output type).
template <typename TI, typename TO>
shared_ptr<op::Constant> fold_constant_convert_helper1(shared_ptr<op::Constant> constant,
                                                       const element::Type& output_element_type)
{
    auto out_shape = constant->get_shape();
    vector<TO> out_vec(shape_size(out_shape));

    runtime::reference::convert<TI, TO>(
        constant->get_vector<TI>().data(), out_vec.data(), shape_size(out_shape));

    return make_shared<op::Constant>(output_element_type, out_shape, out_vec);
}

// Helper for mapping element::Types to runtime::reference::convert, which is templated in C++
// data types. Used by fold_constant_convert, which determines the appropriate C++ type for "TI"
// (input type).
template <typename TI>
shared_ptr<op::Constant> fold_constant_convert_helper0(shared_ptr<op::Constant> constant,
                                                       const element::Type& output_element_type)
{
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (output_element_type.get_type_enum())
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_convert");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_convert");
        break;
    case element::Type_t::boolean:
        return fold_constant_convert_helper1<TI, char>(constant, output_element_type);
    case element::Type_t::bf16:
        return fold_constant_convert_helper1<TI, bfloat16>(constant, output_element_type);
    case element::Type_t::f16:
        return fold_constant_convert_helper1<TI, float16>(constant, output_element_type);
    case element::Type_t::f32:
        return fold_constant_convert_helper1<TI, float>(constant, output_element_type);
    case element::Type_t::f64:
        return fold_constant_convert_helper1<TI, double>(constant, output_element_type);
    case element::Type_t::i8:
        return fold_constant_convert_helper1<TI, int8_t>(constant, output_element_type);
    case element::Type_t::i16:
        return fold_constant_convert_helper1<TI, int16_t>(constant, output_element_type);
    case element::Type_t::i32:
        return fold_constant_convert_helper1<TI, int32_t>(constant, output_element_type);
    case element::Type_t::i64:
        return fold_constant_convert_helper1<TI, int64_t>(constant, output_element_type);
    case element::Type_t::u8:
        return fold_constant_convert_helper1<TI, uint8_t>(constant, output_element_type);
    case element::Type_t::u16:
        return fold_constant_convert_helper1<TI, uint16_t>(constant, output_element_type);
    case element::Type_t::u32:
        return fold_constant_convert_helper1<TI, uint32_t>(constant, output_element_type);
    case element::Type_t::u64:
        return fold_constant_convert_helper1<TI, uint64_t>(constant, output_element_type);
    }
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic pop
#endif
}

static shared_ptr<op::Constant> fold_constant_convert(shared_ptr<op::Constant> constant,
                                                      const element::Type& output_element_type)
{
    auto& input_element_type = constant->get_output_element_type(0);

    if (input_element_type == output_element_type)
    {
        return constant;
    }

#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (input_element_type.get_type_enum())
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_convert");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_convert");
        break;
    case element::Type_t::boolean:
        return fold_constant_convert_helper0<char>(constant, output_element_type);
    case element::Type_t::bf16:
        return fold_constant_convert_helper0<bfloat16>(constant, output_element_type);
    case element::Type_t::f16:
        return fold_constant_convert_helper0<float16>(constant, output_element_type);
    case element::Type_t::f32:
        return fold_constant_convert_helper0<float>(constant, output_element_type);
    case element::Type_t::f64:
        return fold_constant_convert_helper0<double>(constant, output_element_type);
    case element::Type_t::i8:
        return fold_constant_convert_helper0<int8_t>(constant, output_element_type);
    case element::Type_t::i16:
        return fold_constant_convert_helper0<int16_t>(constant, output_element_type);
    case element::Type_t::i32:
        return fold_constant_convert_helper0<int32_t>(constant, output_element_type);
    case element::Type_t::i64:
        return fold_constant_convert_helper0<int64_t>(constant, output_element_type);
    case element::Type_t::u8:
        return fold_constant_convert_helper0<uint8_t>(constant, output_element_type);
    case element::Type_t::u16:
        return fold_constant_convert_helper0<uint16_t>(constant, output_element_type);
    case element::Type_t::u32:
        return fold_constant_convert_helper0<uint32_t>(constant, output_element_type);
    case element::Type_t::u64:
        return fold_constant_convert_helper0<uint64_t>(constant, output_element_type);
    }
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic pop
#endif
}

void pass::ConstantFolding::construct_constant_convert()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::i32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto convert_op = make_shared<op::Convert>(constant_label, element::i64);

    auto constant_convert_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_convert_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto convert_match = static_pointer_cast<op::Convert>(m.get_match_root());

        replace_node(
            m.get_match_root(),
            fold_constant_convert(constant_match, convert_match->get_output_element_type(0)));
        return true;
    };

    auto convert_matcher =
        make_shared<pattern::Matcher>(convert_op, "ConstantFolding.ConstantConvert");
    this->add_matcher(convert_matcher, constant_convert_callback, all_pass_property_off);
}

// ShapeOf is a bit of an odd duck: it doesn't matter if the input's value is
// constant, as long as it has static shape.
void pass::ConstantFolding::construct_constant_shape_of()
{
    auto arg_label = make_shared<pattern::op::Label>(element::i32, Shape{2, 3, 4});
    auto shape_of_op = make_shared<op::ShapeOf>(arg_label);

    auto constant_shape_of_callback = [arg_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_shape_of_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto arg_match = pattern_map[arg_label];

        if (arg_match->get_output_partial_shape(0).is_static())
        {
            auto arg_shape = arg_match->get_output_shape(0);
            auto replacement =
                make_shared<op::Constant>(element::i64, Shape{arg_shape.size()}, arg_shape.data());

            replace_node(m.get_match_root(), replacement);

            return true;
        }
        else
        {
            return false;
        }
    };

    auto shape_of_matcher =
        make_shared<pattern::Matcher>(shape_of_op, "ConstantFolding.ConstantShapeOf");
    this->add_matcher(shape_of_matcher, constant_shape_of_callback, all_pass_property_off);
}

template <typename T>
static shared_ptr<op::Constant> fold_constant_reverse_helper(shared_ptr<op::Constant> constant,
                                                             const AxisSet& reversed_axes)
{
    auto out_shape = constant->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::reverse<T>(
        constant->get_vector<T>().data(), out_vec.data(), out_shape, out_shape, reversed_axes);

    return make_shared<op::Constant>(constant->get_output_element_type(0), out_shape, out_vec);
}

static shared_ptr<op::Constant> fold_constant_reverse(shared_ptr<op::Constant> constant,
                                                      const AxisSet& reversed_axes)
{
    auto& input_element_type = constant->get_output_element_type(0);

#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (input_element_type.get_type_enum())
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_convert");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_convert");
        break;
    case element::Type_t::boolean:
        return fold_constant_reverse_helper<char>(constant, reversed_axes);
    case element::Type_t::bf16:
        return fold_constant_reverse_helper<bfloat16>(constant, reversed_axes);
    case element::Type_t::f16:
        return fold_constant_reverse_helper<float16>(constant, reversed_axes);
    case element::Type_t::f32: return fold_constant_reverse_helper<float>(constant, reversed_axes);
    case element::Type_t::f64: return fold_constant_reverse_helper<double>(constant, reversed_axes);
    case element::Type_t::i8: return fold_constant_reverse_helper<int8_t>(constant, reversed_axes);
    case element::Type_t::i16:
        return fold_constant_reverse_helper<int16_t>(constant, reversed_axes);
    case element::Type_t::i32:
        return fold_constant_reverse_helper<int32_t>(constant, reversed_axes);
    case element::Type_t::i64:
        return fold_constant_reverse_helper<int64_t>(constant, reversed_axes);
    case element::Type_t::u8: return fold_constant_reverse_helper<uint8_t>(constant, reversed_axes);
    case element::Type_t::u16:
        return fold_constant_reverse_helper<uint16_t>(constant, reversed_axes);
    case element::Type_t::u32:
        return fold_constant_reverse_helper<uint32_t>(constant, reversed_axes);
    case element::Type_t::u64:
        return fold_constant_reverse_helper<uint64_t>(constant, reversed_axes);
    }
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic pop
#endif
}

void pass::ConstantFolding::construct_constant_reverse()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::i32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto convert_op = make_shared<op::Reverse>(constant_label, AxisSet{0, 1, 2});

    auto constant_reverse_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_reverse_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto reverse_match = static_pointer_cast<op::Reverse>(m.get_match_root());

        replace_node(m.get_match_root(),
                     fold_constant_reverse(constant_match, reverse_match->get_reversed_axes()));
        return true;
    };

    auto convert_matcher =
        make_shared<pattern::Matcher>(convert_op, "ConstantFolding.ConstantReverse");
    this->add_matcher(convert_matcher, constant_reverse_callback, all_pass_property_off);
}

template <typename T>
static shared_ptr<op::Constant> fold_constant_product_helper(shared_ptr<op::Constant> constant,
                                                             const AxisSet& reduction_axes,
                                                             const Shape& result_shape)
{
    vector<T> out_vec(shape_size(result_shape));

    runtime::reference::product<T>(constant->get_vector<T>().data(),
                                   out_vec.data(),
                                   constant->get_output_shape(0),
                                   result_shape,
                                   reduction_axes);

    return make_shared<op::Constant>(constant->get_output_element_type(0), result_shape, out_vec);
}

static shared_ptr<op::Constant> fold_constant_product(shared_ptr<op::Constant> constant,
                                                      const AxisSet& reduction_axes,
                                                      const Shape& result_shape)
{
    auto& input_element_type = constant->get_output_element_type(0);

#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (input_element_type.get_type_enum())
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_product");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_product");
        break;
    case element::Type_t::boolean:
        return fold_constant_product_helper<char>(constant, reduction_axes, result_shape);
    case element::Type_t::bf16:
        return fold_constant_product_helper<bfloat16>(constant, reduction_axes, result_shape);
    case element::Type_t::f16:
        return fold_constant_product_helper<float16>(constant, reduction_axes, result_shape);
    case element::Type_t::f32:
        return fold_constant_product_helper<float>(constant, reduction_axes, result_shape);
    case element::Type_t::f64:
        return fold_constant_product_helper<double>(constant, reduction_axes, result_shape);
    case element::Type_t::i8:
        return fold_constant_product_helper<int8_t>(constant, reduction_axes, result_shape);
    case element::Type_t::i16:
        return fold_constant_product_helper<int16_t>(constant, reduction_axes, result_shape);
    case element::Type_t::i32:
        return fold_constant_product_helper<int32_t>(constant, reduction_axes, result_shape);
    case element::Type_t::i64:
        return fold_constant_product_helper<int64_t>(constant, reduction_axes, result_shape);
    case element::Type_t::u8:
        return fold_constant_product_helper<uint8_t>(constant, reduction_axes, result_shape);
    case element::Type_t::u16:
        return fold_constant_product_helper<uint16_t>(constant, reduction_axes, result_shape);
    case element::Type_t::u32:
        return fold_constant_product_helper<uint32_t>(constant, reduction_axes, result_shape);
    case element::Type_t::u64:
        return fold_constant_product_helper<uint64_t>(constant, reduction_axes, result_shape);
    }
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic pop
#endif
}

void pass::ConstantFolding::construct_constant_product()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::i32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto convert_op = make_shared<op::Product>(constant_label, AxisSet{0, 1, 2});

    auto constant_product_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_product_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto product_match = static_pointer_cast<op::Product>(m.get_match_root());

        replace_node(m.get_match_root(),
                     fold_constant_product(constant_match,
                                           product_match->get_reduction_axes(),
                                           product_match->get_output_shape(0)));
        return true;
    };

    auto convert_matcher =
        make_shared<pattern::Matcher>(convert_op, "ConstantFolding.ConstantProduct");
    this->add_matcher(convert_matcher, constant_product_callback, all_pass_property_off);
}
