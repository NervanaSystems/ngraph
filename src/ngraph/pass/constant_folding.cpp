//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/maximum.hpp"
#include "ngraph/runtime/reference/minimum.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/negate.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/quantize.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/subtract.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> make_constant_reshape(shared_ptr<op::Constant> constant,
                                               shared_ptr<op::Reshape> reshape)
{
    auto out_shape = reshape->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::reshape<T>(constant->get_vector<T>().data(),
                                   out_vec.data(),
                                   constant->get_shape(),
                                   reshape->get_input_order(),
                                   out_shape);

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

template <class T>
shared_ptr<op::Constant> make_constant_pad(shared_ptr<op::Constant> constant,
                                           shared_ptr<op::Pad> pad)
{
    auto out_shape = pad->get_shape();
    vector<T> out_vec(shape_size(out_shape));
    auto pad_value = std::static_pointer_cast<op::Constant>(pad->get_argument(1));

    runtime::reference::pad<T>(constant->get_vector<T>().data(),
                               pad_value->get_vector<T>().data(),
                               out_vec.data(),
                               constant->get_shape(),
                               out_shape,
                               pad->get_padding_below(),
                               pad->get_padding_above(),
                               pad->get_padding_interior());

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void ngraph::pass::ConstantFolding::construct_constant_pad()
{
    auto is_constant = pattern::has_class<op::Constant>();
    auto constant_label = make_shared<pattern::op::Label>(element::f32, Shape{6}, is_constant);

    auto pad_value_label = make_shared<pattern::op::Label>(element::f32, Shape{}, is_constant);

    Shape padding_below{0};
    Shape padding_above{0};
    Shape padding_interior{0};

    auto pad = make_shared<op::Pad>(
        constant_label, pad_value_label, padding_below, padding_above, padding_interior);

    auto constant_pad_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_pad_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto pad_match = static_pointer_cast<op::Pad>(m.get_match_root());

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(), make_constant_pad<int>(constant_match, pad_match));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(), make_constant_pad<int8_t>(constant_match, pad_match));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(), make_constant_pad<float>(constant_match, pad_match));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(), make_constant_pad<double>(constant_match, pad_match));
            return true;
        }

        return false;
    };

    auto pad_matcher = make_shared<pattern::Matcher>(pad, constant_pad_callback);
    this->add_matcher(pad_matcher);
}

void ngraph::pass::ConstantFolding::construct_constant_reshape()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto reshape = make_shared<op::Reshape>(constant_label, AxisVector{0, 1}, Shape{2, 4, 1});

    auto constant_reshape_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_reshape_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto reshape_match = static_pointer_cast<op::Reshape>(m.get_match_root());

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         make_constant_reshape<int>(constant_match, reshape_match));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         make_constant_reshape<int8_t>(constant_match, reshape_match));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         make_constant_reshape<float>(constant_match, reshape_match));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         make_constant_reshape<double>(constant_match, reshape_match));
            return true;
        }

        return false;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(reshape, constant_reshape_callback);
    this->add_matcher(reshape_matcher);
}

template <class T>
shared_ptr<op::Constant> make_constant_broadcast(shared_ptr<op::Constant> constant,
                                                 shared_ptr<op::Broadcast> broadcast)
{
    auto out_shape = broadcast->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::broadcast<T>(constant->get_vector<T>().data(),
                                     out_vec.data(),
                                     constant->get_shape(),
                                     out_shape,
                                     broadcast->get_broadcast_axes());

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void ngraph::pass::ConstantFolding::construct_constant_broadcast()
{
    auto constant_label =
        make_shared<pattern::op::Label>(element::f32, Shape{2}, pattern::has_class<op::Constant>());

    auto broadcast = make_shared<op::Broadcast>(constant_label, Shape{2, 4}, AxisSet{1});

    auto constant_broadcast_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_broadcast_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto broadcast_match = static_pointer_cast<op::Broadcast>(m.get_match_root());

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         make_constant_broadcast<int>(constant_match, broadcast_match));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         make_constant_broadcast<int8_t>(constant_match, broadcast_match));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         make_constant_broadcast<float>(constant_match, broadcast_match));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         make_constant_broadcast<double>(constant_match, broadcast_match));
            return true;
        }

        return false;
    };

    auto broadcast_matcher = make_shared<pattern::Matcher>(broadcast, constant_broadcast_callback);
    this->add_matcher(broadcast_matcher);
}

template <class T>
shared_ptr<op::Constant> make_constant_binary(shared_ptr<op::Constant> a,
                                              shared_ptr<op::Constant> b,
                                              shared_ptr<Node> binary)
{
    auto out_shape = binary->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    if (std::dynamic_pointer_cast<op::Add>(binary))
    {
        runtime::reference::add<T>(a->get_vector<T>().data(),
                                   b->get_vector<T>().data(),
                                   out_vec.data(),
                                   shape_size(out_shape));
    }
    else if (std::dynamic_pointer_cast<op::Subtract>(binary))
    {
        runtime::reference::subtract<T>(a->get_vector<T>().data(),
                                        b->get_vector<T>().data(),
                                        out_vec.data(),
                                        shape_size(out_shape));
    }
    else if (std::dynamic_pointer_cast<op::Multiply>(binary))
    {
        runtime::reference::multiply<T>(a->get_vector<T>().data(),
                                        b->get_vector<T>().data(),
                                        out_vec.data(),
                                        shape_size(out_shape));
    }
    else if (std::dynamic_pointer_cast<op::Divide>(binary))
    {
        runtime::reference::divide<T>(a->get_vector<T>().data(),
                                      b->get_vector<T>().data(),
                                      out_vec.data(),
                                      shape_size(out_shape));
    }
    else if (std::dynamic_pointer_cast<op::Minimum>(binary))
    {
        runtime::reference::minimum<T>(a->get_vector<T>().data(),
                                       b->get_vector<T>().data(),
                                       out_vec.data(),
                                       shape_size(out_shape));
    }
    else if (std::dynamic_pointer_cast<op::Maximum>(binary))
    {
        runtime::reference::maximum<T>(a->get_vector<T>().data(),
                                       b->get_vector<T>().data(),
                                       out_vec.data(),
                                       shape_size(out_shape));
    }
    else
    {
        NGRAPH_ASSERT(false)
            << "make_constant_binary must be consistent with is_supported_binary_op";
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

void ngraph::pass::ConstantFolding::construct_constant_binary()
{
    auto a = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto b = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto is_bea = pattern::has_class<op::util::BinaryElementwiseArithmetic>();
    auto bea = std::make_shared<pattern::op::Any>(a, is_bea, NodeVector{a, b});

    auto constant_binary_callback = [a, b](pattern::Matcher& m) {
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

        auto type = a_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(),
                         make_constant_binary<int>(a_match, b_match, binary_match));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         make_constant_binary<int8_t>(a_match, b_match, binary_match));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         make_constant_binary<float>(a_match, b_match, binary_match));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         make_constant_binary<double>(a_match, b_match, binary_match));
            return true;
        }

        return false;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(bea, constant_binary_callback);
    this->add_matcher(reshape_matcher);
}

bool is_supported_unary_op(std::shared_ptr<Node> n)
{
    return std::dynamic_pointer_cast<op::Abs>(n) || std::dynamic_pointer_cast<op::Negative>(n);
}

template <class T>
shared_ptr<op::Constant> make_constant_unary(shared_ptr<op::Constant> constant,
                                             shared_ptr<Node> unary)
{
    auto out_shape = unary->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    if (std::dynamic_pointer_cast<op::Abs>(unary))
    {
        runtime::reference::abs<T>(
            constant->get_vector<T>().data(), out_vec.data(), shape_size(out_shape));
    }
    else if (std::dynamic_pointer_cast<op::Negative>(unary))
    {
        runtime::reference::negate<T>(
            constant->get_vector<T>().data(), out_vec.data(), shape_size(out_shape));
    }
    else
    {
        NGRAPH_ASSERT(false) << "must be consistent with is_supported_unary_op";
    }

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void ngraph::pass::ConstantFolding::construct_constant_unary()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto is_uea = pattern::has_class<op::util::UnaryElementwiseArithmetic>();
    auto uea =
        std::make_shared<pattern::op::Any>(constant_label, is_uea, NodeVector{constant_label});

    auto constant_unary_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_reshape_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = dynamic_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto unary_match = m.get_match_root();

        if (!is_supported_unary_op(unary_match))
        {
            return false;
        }

        auto type = constant_match->get_element_type();
        if (type == element::i32)
        {
            replace_node(m.get_match_root(), make_constant_unary<int>(constant_match, unary_match));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         make_constant_unary<int8_t>(constant_match, unary_match));
            return true;
        }
        else if (type == element::f32)
        {
            replace_node(m.get_match_root(),
                         make_constant_unary<float>(constant_match, unary_match));
            return true;
        }
        else if (type == element::f64)
        {
            replace_node(m.get_match_root(),
                         make_constant_unary<double>(constant_match, unary_match));
            return true;
        }

        return false;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(uea, constant_unary_callback);
    this->add_matcher(reshape_matcher);
}

template <class QUANT, class REAL>
shared_ptr<op::Constant> make_constant_dequantize(shared_ptr<op::Constant> constant,
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

void ngraph::pass::ConstantFolding::construct_constant_dequantize()
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
                         make_constant_dequantize<uint8_t, float>(
                             constant_match, dequantize_op, scale, offset));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         make_constant_dequantize<int8_t, float>(
                             constant_match, dequantize_op, scale, offset));
            return true;
        }

        return false;
    };

    auto dequantize_matcher = make_shared<pattern::Matcher>(dequant, constant_dequantize_callback);
    this->add_matcher(dequantize_matcher);
}

template <class REAL, class QUANT>
shared_ptr<op::Constant> make_constant_quantize(shared_ptr<op::Constant> constant,
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

void ngraph::pass::ConstantFolding::construct_constant_quantize()
{
    auto constant_label =
        make_shared<pattern::op::Label>(element::f32, Shape{2}, pattern::has_class<op::Constant>());
    auto q_scale = op::Constant::create(element::f32, Shape{}, {1});
    auto q_offset = op::Constant::create(element::i8, Shape{}, {0});
    auto mode = op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO;
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

        if (quantize_op->get_round_mode() != op::Quantize::RoundMode::HALF_AWAY_FROM_ZERO)
        {
            return false;
        }

        if (type == element::u8)
        {
            replace_node(
                m.get_match_root(),
                make_constant_quantize<float, uint8_t>(constant_match, quantize_op, scale, offset));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(
                m.get_match_root(),
                make_constant_quantize<float, int8_t>(constant_match, quantize_op, scale, offset));
            return true;
        }

        return false;
    };

    auto quantize_matcher = make_shared<pattern::Matcher>(quant, constant_quantize_callback);
    this->add_matcher(quantize_matcher);
}
