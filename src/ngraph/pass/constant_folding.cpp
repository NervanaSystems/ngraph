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
#include <stdint.h>

#include "constant_folding.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/all.hpp"
#include "ngraph/runtime/reference/and.hpp"
#include "ngraph/runtime/reference/any.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/ceiling.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/convert.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/equal.hpp"
#include "ngraph/runtime/reference/floor.hpp"
#include "ngraph/runtime/reference/gather.hpp"
#include "ngraph/runtime/reference/greater.hpp"
#include "ngraph/runtime/reference/greater_eq.hpp"
#include "ngraph/runtime/reference/less.hpp"
#include "ngraph/runtime/reference/less_eq.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/maximum.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/minimum.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/negate.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/not_equal.hpp"
#include "ngraph/runtime/reference/or.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/quantize.hpp"
#include "ngraph/runtime/reference/range.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/select.hpp"
#include "ngraph/runtime/reference/sign.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/runtime/reference/sqrt.hpp"
#include "ngraph/runtime/reference/subtract.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/runtime/reference/xor.hpp"
#include "ngraph/slice_plan.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static bool revalidate_and_ensure_static(shared_ptr<Node> n)
{
    n->revalidate_and_infer_types();
    for (auto& o : n->outputs())
    {
        if (o.get_partial_shape().is_dynamic() || o.get_element_type().is_dynamic())
        {
            return false;
        }
    }
    return true;
}

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
        switch (type.get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_reshape_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_reshape_callback");
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

template <class T>
shared_ptr<op::Constant> fold_constant_pad(shared_ptr<op::Constant> constant,
                                           shared_ptr<op::Pad> pad,
                                           NodeExecutorTy func)
{
    auto out_shape = pad->get_shape();
    vector<T> out_vec(shape_size(out_shape));
    auto pad_value = std::static_pointer_cast<op::Constant>(
        pad->input(1).get_source_output().get_node_shared_ptr());

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

        NGRAPH_CHECK(revalidate_and_ensure_static(pad_match));

        NodeExecutorTy func = nullptr;
        if (!m_cfmap.empty())
        {
            auto handler = m_cfmap.find(type_index(typeid(ngraph::op::Pad)));
            NGRAPH_CHECK(handler != m_cfmap.end(), "constant folding map should have pad entry");
            func = handler->second(pad_match.get());
        }

        std::shared_ptr<Node> replacement;
        auto type = constant_match->get_element_type();
        switch (type.get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_pad_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_pad_callback");
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

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto pad_matcher = make_shared<pattern::Matcher>(pad, "ConstantFolding.ConstantPad");
    this->add_matcher(pad_matcher, constant_pad_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

template <class T>
shared_ptr<op::Constant> fold_constant_dyn_reshape(shared_ptr<op::Constant> constant_data,
                                                   shared_ptr<op::DynReshape> dyn_reshape)
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
    auto dyn_reshape = make_shared<op::DynReshape>(constant_data_label, constant_shape_label);

    // Note: No need to capture or consider constant_shape_label, because
    // shape propagation will have transferred the info to dyn_reshape's
    // output.
    auto constant_dyn_reshape_callback = [constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_dyn_reshape_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_data_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto dyn_reshape_match = static_pointer_cast<op::DynReshape>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(dyn_reshape_match));

        std::shared_ptr<Node> replacement;
        auto type = dyn_reshape_match->get_element_type();
        switch (type.get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_dyn_reshape_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false,
                         "Encountered 'dynamic' element type in constant_dyn_reshape_callback");
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

template <class T>
shared_ptr<op::Constant> fold_constant_transpose(shared_ptr<op::Constant> constant_data,
                                                 shared_ptr<op::Constant> constant_perm,
                                                 shared_ptr<op::Transpose> transpose)
{
    auto out_shape = transpose->get_shape();
    auto input_order = constant_perm->get_axis_vector_val();

    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::reshape<T>(constant_data->get_data_ptr<T>(),
                                   out_vec.data(),
                                   constant_data->get_shape(),
                                   input_order,
                                   out_shape);

    return make_shared<op::Constant>(transpose->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_transpose()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto constant_perm_label =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto transpose = make_shared<op::Transpose>(constant_data_label, constant_perm_label);

    auto constant_transpose_callback = [constant_data_label,
                                        constant_perm_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_transpose_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_data_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto constant_perm_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_perm_label]);
        auto transpose_match = static_pointer_cast<op::Transpose>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(transpose_match));

        std::shared_ptr<Node> replacement;
        auto type = transpose_match->get_element_type();
        switch (type.get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_transpose_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false,
                         "Encountered 'dynamic' element type in constant_transpose_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_transpose<char>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_transpose<bfloat16>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_transpose<float16>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_transpose<float>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_transpose<double>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_transpose<int8_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_transpose<int16_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_transpose<int32_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_transpose<int64_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_transpose<uint8_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_transpose<uint16_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_transpose<uint32_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_transpose<uint64_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto transpose_matcher =
        make_shared<pattern::Matcher>(transpose, "ConstantFolding.ConstantTranspose");
    this->add_matcher(
        transpose_matcher, constant_transpose_callback, PassProperty::CHANGE_DYNAMIC_STATE);
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
        switch (type.get_type_enum())
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

    auto broadcast_matcher =
        make_shared<pattern::Matcher>(broadcast, "ConstantFolding.ConstantBroadcast");
    this->add_matcher(
        broadcast_matcher, constant_broadcast_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

template <class T>
shared_ptr<op::Constant> fold_constant_dyn_broadcast(shared_ptr<op::Constant> arg,
                                                     shared_ptr<op::Constant> shape,
                                                     shared_ptr<op::Constant> axes)
{
    auto out_shape = shape->get_shape_val();
    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::broadcast<T>(arg->get_data_ptr<T>(),
                                     out_vec.data(),
                                     arg->get_shape(),
                                     out_shape,
                                     axes->get_axis_set_val());

    return make_shared<op::Constant>(arg->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_dyn_broadcast()
{
    auto constant_arg_label =
        make_shared<pattern::op::Label>(element::f32, Shape{2}, pattern::has_class<op::Constant>());
    auto constant_shape_label =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto constant_axes_label =
        make_shared<pattern::op::Label>(element::i64, Shape{1}, pattern::has_class<op::Constant>());

    auto dyn_broadcast = make_shared<op::DynBroadcast>(
        constant_arg_label, constant_shape_label, constant_axes_label);

    auto constant_dyn_broadcast_callback = [constant_arg_label,
                                            constant_shape_label,
                                            constant_axes_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_dyn_broadcast_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_arg_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_arg_label]);
        auto constant_shape_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_shape_label]);
        auto constant_axes_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_axes_label]);
        auto dyn_broadcast_match = static_pointer_cast<op::DynBroadcast>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(dyn_broadcast_match));

        std::shared_ptr<Node> replacement;
        auto type = dyn_broadcast_match->get_output_element_type(0);
        switch (type.get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_dyn_broadcast_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false,
                         "Encountered 'dynamic' element type in constant_dyn_broadcast_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_dyn_broadcast<char>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_dyn_broadcast<bfloat16>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_dyn_broadcast<float16>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_dyn_broadcast<float>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_dyn_broadcast<double>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_dyn_broadcast<int8_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_dyn_broadcast<int16_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_dyn_broadcast<int32_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_dyn_broadcast<int64_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_dyn_broadcast<uint8_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_dyn_broadcast<uint16_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_dyn_broadcast<uint32_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_dyn_broadcast<uint64_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto dyn_broadcast_matcher =
        make_shared<pattern::Matcher>(dyn_broadcast, "ConstantFolding.ConstantDynBroadcast");
    this->add_matcher(
        dyn_broadcast_matcher, constant_dyn_broadcast_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

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
    switch (et_out.get_type_enum())
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
        switch (in_type.get_type_enum())
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

bool is_supported_unary_op(std::shared_ptr<Node> n)
{
    return std::dynamic_pointer_cast<op::Abs>(n) || std::dynamic_pointer_cast<op::Ceiling>(n) ||
           std::dynamic_pointer_cast<op::Floor>(n) || std::dynamic_pointer_cast<op::Negative>(n) ||
           std::dynamic_pointer_cast<op::Not>(n) || std::dynamic_pointer_cast<op::Relu>(n) ||
           std::dynamic_pointer_cast<op::Sign>(n) || std::dynamic_pointer_cast<op::Sqrt>(n);
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
        if (std::dynamic_pointer_cast<op::Abs>(unary))
        {
            runtime::reference::abs<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Ceiling>(unary))
        {
            runtime::reference::ceiling<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Floor>(unary))
        {
            runtime::reference::floor<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Negative>(unary))
        {
            runtime::reference::negate<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Not>(unary))
        {
            runtime::reference::logical_not<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Relu>(unary))
        {
            runtime::reference::relu<T>(
                constant->get_data_ptr<T>(), out_vec.data(), shape_size(out_shape));
        }
        else if (std::dynamic_pointer_cast<op::Sign>(unary))
        {
            runtime::reference::sign<T>(
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
    auto is_ue = [](std::shared_ptr<Node> n) {
        return (pattern::has_class<op::util::UnaryElementwiseArithmetic>()(n) ||
                pattern::has_class<op::Not>()(n));
    };
    auto ue = std::make_shared<pattern::op::Any>(constant_label, is_ue, NodeVector{constant_label});

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
        switch (type.get_type_enum())
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

        auto scale = dynamic_pointer_cast<op::Constant>(
            dequant_match->input(1).get_source_output().get_node_shared_ptr());
        auto offset = dynamic_pointer_cast<op::Constant>(
            dequant_match->input(2).get_source_output().get_node_shared_ptr());

        NGRAPH_CHECK(revalidate_and_ensure_static(dequantize_op));
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
        dequantize_matcher, constant_dequantize_callback, PassProperty::CHANGE_DYNAMIC_STATE);
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

        NGRAPH_CHECK(revalidate_and_ensure_static(quantize_op));

        auto args = quant_match->get_arguments();
        auto scale = static_pointer_cast<op::Constant>(
            quant_match->input(1).get_source_output().get_node_shared_ptr());
        auto offset = static_pointer_cast<op::Constant>(
            quant_match->input(2).get_source_output().get_node_shared_ptr());

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
        quantize_matcher, constant_quantize_callback, PassProperty::CHANGE_DYNAMIC_STATE);
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
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
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

    NGRAPH_UNREACHABLE("Unexpected switch case");
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
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

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
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

    NGRAPH_UNREACHABLE("Unexpected switch case");
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
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

        NGRAPH_CHECK(revalidate_and_ensure_static(convert_match));

        replace_node(
            m.get_match_root(),
            fold_constant_convert(constant_match, convert_match->get_output_element_type(0)));
        return true;
    };

    auto convert_matcher =
        make_shared<pattern::Matcher>(convert_op, "ConstantFolding.ConstantConvert");
    this->add_matcher(
        convert_matcher, constant_convert_callback, PassProperty::CHANGE_DYNAMIC_STATE);
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
            NGRAPH_CHECK(revalidate_and_ensure_static(m.get_match_root()));

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
    this->add_matcher(
        shape_of_matcher, constant_shape_of_callback, PassProperty::CHANGE_DYNAMIC_STATE);
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

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
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

    NGRAPH_UNREACHABLE("Unexpected switch case");

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
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

        NGRAPH_CHECK(revalidate_and_ensure_static(reverse_match));

        replace_node(m.get_match_root(),
                     fold_constant_reverse(constant_match, reverse_match->get_reversed_axes()));
        return true;
    };

    auto convert_matcher =
        make_shared<pattern::Matcher>(convert_op, "ConstantFolding.ConstantReverse");
    this->add_matcher(
        convert_matcher, constant_reverse_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

template <typename T>
static shared_ptr<op::Constant>
    fold_constant_arithmetic_reduction_helper(shared_ptr<op::Constant> constant,
                                              shared_ptr<Node> reduction_node)
{
    vector<T> out_vec(shape_size(reduction_node->get_shape()));

    if (auto max = dynamic_pointer_cast<op::Max>(reduction_node))
    {
        runtime::reference::max<T>(constant->get_vector<T>().data(),
                                   out_vec.data(),
                                   constant->get_output_shape(0),
                                   reduction_node->get_shape(),
                                   max->get_reduction_axes());
    }
    else if (auto min = dynamic_pointer_cast<op::Min>(reduction_node))
    {
        runtime::reference::min<T>(constant->get_vector<T>().data(),
                                   out_vec.data(),
                                   constant->get_output_shape(0),
                                   reduction_node->get_shape(),
                                   min->get_reduction_axes());
    }
    else if (auto prod = dynamic_pointer_cast<op::Product>(reduction_node))
    {
        runtime::reference::product<T>(constant->get_vector<T>().data(),
                                       out_vec.data(),
                                       constant->get_output_shape(0),
                                       reduction_node->get_shape(),
                                       prod->get_reduction_axes());
    }
    else if (auto sum = dynamic_pointer_cast<op::Sum>(reduction_node))
    {
        runtime::reference::sum<T>(constant->get_vector<T>().data(),
                                   out_vec.data(),
                                   constant->get_output_shape(0),
                                   reduction_node->get_shape(),
                                   sum->get_reduction_axes());
    }
    else
    {
        NGRAPH_CHECK(false,
                     "Internal nGraph error: Ops handled in "
                     "fold_constant_arithmetic_reduction_helper must be consistent with those "
                     "matched in construct_constant_arithmetic_reduction");
    }

    return make_shared<op::Constant>(
        reduction_node->get_output_element_type(0), reduction_node->get_shape(), out_vec);
}

static shared_ptr<op::Constant>
    fold_constant_arithmetic_reduction(shared_ptr<op::Constant> constant,
                                       shared_ptr<Node> reduction_node)
{
    auto& input_element_type = constant->get_output_element_type(0);

    switch (input_element_type.get_type_enum())
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false,
                     "Encountered 'undefined' element type in fold_constant_arithmetic_reduction");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false,
                     "Encountered 'dynamic' element type in fold_constant_arithmetic_reduction");
        break;
    case element::Type_t::boolean:
        return fold_constant_arithmetic_reduction_helper<char>(constant, reduction_node);
    case element::Type_t::bf16:
        return fold_constant_arithmetic_reduction_helper<bfloat16>(constant, reduction_node);
    case element::Type_t::f16:
        return fold_constant_arithmetic_reduction_helper<float16>(constant, reduction_node);
    case element::Type_t::f32:
        return fold_constant_arithmetic_reduction_helper<float>(constant, reduction_node);
    case element::Type_t::f64:
        return fold_constant_arithmetic_reduction_helper<double>(constant, reduction_node);
    case element::Type_t::i8:
        return fold_constant_arithmetic_reduction_helper<int8_t>(constant, reduction_node);
    case element::Type_t::i16:
        return fold_constant_arithmetic_reduction_helper<int16_t>(constant, reduction_node);
    case element::Type_t::i32:
        return fold_constant_arithmetic_reduction_helper<int32_t>(constant, reduction_node);
    case element::Type_t::i64:
        return fold_constant_arithmetic_reduction_helper<int64_t>(constant, reduction_node);
    case element::Type_t::u8:
        return fold_constant_arithmetic_reduction_helper<uint8_t>(constant, reduction_node);
    case element::Type_t::u16:
        return fold_constant_arithmetic_reduction_helper<uint16_t>(constant, reduction_node);
    case element::Type_t::u32:
        return fold_constant_arithmetic_reduction_helper<uint32_t>(constant, reduction_node);
    case element::Type_t::u64:
        return fold_constant_arithmetic_reduction_helper<uint64_t>(constant, reduction_node);
    }

    NGRAPH_UNREACHABLE("Unexpected switch case");
}

void pass::ConstantFolding::construct_constant_arithmetic_reduction()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::i32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto constant_axes_label =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto is_supported_reduction = [](std::shared_ptr<Node> n) {
        return (pattern::has_class<op::Max>()(n) || pattern::has_class<op::Min>()(n) ||
                pattern::has_class<op::Product>()(n) || pattern::has_class<op::Sum>()(n));
    };
    auto reduction =
        std::make_shared<pattern::op::Any>(element::i32,
                                           Shape{2},
                                           is_supported_reduction,
                                           NodeVector{constant_data_label, constant_axes_label});

    auto constant_arithmetic_reduction_callback = [constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_arithmetic_reduction_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto reduction_match = m.get_match_root();

        NGRAPH_CHECK(revalidate_and_ensure_static(reduction_match));

        replace_node(reduction_match,
                     fold_constant_arithmetic_reduction(constant_match, reduction_match));
        return true;
    };

    auto arithmetic_reduction_matcher =
        make_shared<pattern::Matcher>(reduction, "ConstantFolding.ConstantArithmeticReduction");
    this->add_matcher(arithmetic_reduction_matcher,
                      constant_arithmetic_reduction_callback,
                      PassProperty::CHANGE_DYNAMIC_STATE);
}

static shared_ptr<op::Constant> fold_constant_logical_reduction(shared_ptr<op::Constant> constant,
                                                                shared_ptr<Node> reduction_node)
{
    vector<char> out_vec(shape_size(reduction_node->get_shape()));

    if (auto all = dynamic_pointer_cast<::ngraph::op::All>(reduction_node))
    {
        runtime::reference::all(constant->get_vector<char>().data(),
                                out_vec.data(),
                                constant->get_output_shape(0),
                                reduction_node->get_shape(),
                                all->get_reduction_axes());
    }
    else if (auto any = dynamic_pointer_cast<::ngraph::op::Any>(reduction_node))
    {
        runtime::reference::any(constant->get_vector<char>().data(),
                                out_vec.data(),
                                constant->get_output_shape(0),
                                reduction_node->get_shape(),
                                any->get_reduction_axes());
    }
    else
    {
        NGRAPH_CHECK(false,
                     "Internal nGraph error: Ops handled in "
                     "fold_constant_logical_reduction must be consistent with those "
                     "matched in construct_constant_logical_reduction");
    }

    return make_shared<op::Constant>(
        reduction_node->get_output_element_type(0), reduction_node->get_shape(), out_vec);
}

void pass::ConstantFolding::construct_constant_logical_reduction()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::boolean, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto constant_axes_label =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto is_supported_reduction = [](std::shared_ptr<Node> n) {
        return (pattern::has_class<::ngraph::op::All>()(n) ||
                pattern::has_class<::ngraph::op::Any>()(n));
    };
    auto reduction =
        std::make_shared<pattern::op::Any>(element::i32,
                                           Shape{2},
                                           is_supported_reduction,
                                           NodeVector{constant_data_label, constant_axes_label});

    auto constant_logical_reduction_callback = [constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_logical_reduction_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto reduction_match = m.get_match_root();

        NGRAPH_CHECK(revalidate_and_ensure_static(reduction_match));

        replace_node(reduction_match,
                     fold_constant_logical_reduction(constant_match, reduction_match));
        return true;
    };

    auto logical_reduction_matcher =
        make_shared<pattern::Matcher>(reduction, "ConstantFolding.ConstantLogicalReduction");
    this->add_matcher(logical_reduction_matcher,
                      constant_logical_reduction_callback,
                      PassProperty::CHANGE_DYNAMIC_STATE);
}

template <typename T>
static shared_ptr<op::Constant> fold_constant_concat_helper(const shared_ptr<op::Concat>& concat)
{
    auto concat_inputs = concat->inputs();
    std::vector<const T*> arg_bufs;
    std::vector<Shape> arg_shapes;

    for (auto& input : concat_inputs)
    {
        auto k = static_cast<op::Constant*>(input.get_source_output().get_node());
        arg_bufs.push_back(k->get_data_ptr<T>());
        arg_shapes.push_back(input.get_shape());
    }

    std::vector<T> result_vec(shape_size(concat->get_shape()));

    runtime::reference::concat<T>(arg_bufs,
                                  result_vec.data(),
                                  arg_shapes,
                                  concat->get_shape(),
                                  concat->get_concatenation_axis());

    return make_shared<op::Constant>(
        concat->get_output_element_type(0), concat->get_output_shape(0), result_vec);
}

void pass::ConstantFolding::construct_constant_concat()
{
    auto concat_op = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Concat>());

    auto constant_concat_callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_concat_callback against node = "
                     << m.get_match_root()->get_name();

        auto concat_node = static_pointer_cast<op::Concat>(m.get_match_root());
        auto concat_inputs = concat_node->inputs();

        if (std::any_of(concat_inputs.begin(), concat_inputs.end(), [](const Input<Node>& input) {
                return !(input.get_source_output().get_node()->is_constant());
            }))
        {
            return false;
        }

        NGRAPH_CHECK(revalidate_and_ensure_static(concat_node));

        std::shared_ptr<op::Constant> replacement;

        switch (concat_node->get_output_element_type(0).get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_concat");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_concat");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_concat_helper<char>(concat_node);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_concat_helper<bfloat16>(concat_node);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_concat_helper<float16>(concat_node);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_concat_helper<float>(concat_node);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_concat_helper<double>(concat_node);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_concat_helper<int8_t>(concat_node);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_concat_helper<int16_t>(concat_node);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_concat_helper<int32_t>(concat_node);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_concat_helper<int64_t>(concat_node);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_concat_helper<uint8_t>(concat_node);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_concat_helper<uint16_t>(concat_node);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_concat_helper<uint32_t>(concat_node);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_concat_helper<uint64_t>(concat_node);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto concat_matcher =
        make_shared<pattern::Matcher>(concat_op, "ConstantFolding.ConstantConcat");
    this->add_matcher(concat_matcher, constant_concat_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

// "Inner" helper for fold_constant_gather, which has to switch on the indices
// element type.
template <typename T, typename U>
static shared_ptr<op::Constant> fold_constant_gather_helper(const shared_ptr<op::Constant>& data,
                                                            const shared_ptr<op::Constant>& indices,
                                                            const shared_ptr<op::Gather>& gather)
{
    std::vector<T> result_vec(shape_size(gather->get_shape()));

    runtime::reference::gather<T, U>(data->get_data_ptr<T>(),
                                     indices->get_data_ptr<U>(),
                                     result_vec.data(),
                                     data->get_shape(),
                                     indices->get_shape(),
                                     gather->get_shape(),
                                     gather->get_axis());

    return make_shared<op::Constant>(
        gather->get_output_element_type(0), gather->get_output_shape(0), result_vec);
}

template <typename T>
static shared_ptr<op::Constant> fold_constant_gather(const shared_ptr<op::Constant>& data,
                                                     const shared_ptr<op::Constant>& indices,
                                                     const shared_ptr<op::Gather>& gather)
{
    auto indices_type = indices->get_output_element_type(0);

    switch (indices_type.get_type_enum())
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_gather_callback");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_gather_callback");
        break;
    case element::Type_t::boolean:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::f32:
    case element::Type_t::f64:
    case element::Type_t::i8:
    case element::Type_t::i16:
    case element::Type_t::u8:
    case element::Type_t::u16:
    case element::Type_t::u32:
    case element::Type_t::u64:
        NGRAPH_CHECK(false,
                     "Encountered unsupported indices element type in constant_gather_callback: ",
                     indices_type);
        break;
    case element::Type_t::i32:
        return fold_constant_gather_helper<T, int32_t>(data, indices, gather);
    case element::Type_t::i64:
        return fold_constant_gather_helper<T, int64_t>(data, indices, gather);
    }

    NGRAPH_UNREACHABLE("Unhandled switch case");
}

void pass::ConstantFolding::construct_constant_gather()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{10, 20, 30}, pattern::has_class<op::Constant>());
    auto indices_label =
        make_shared<pattern::op::Label>(element::i64, Shape{5}, pattern::has_class<op::Constant>());
    size_t gather_axis = 1;
    auto gather_op = make_shared<op::Gather>(data_label, indices_label, gather_axis);

    auto constant_gather_callback = [data_label, indices_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_gather_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto data = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        auto indices = static_pointer_cast<op::Constant>(pattern_map[indices_label]);
        auto gather = static_pointer_cast<op::Gather>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(gather));

        std::shared_ptr<Node> replacement;
        auto data_type = data->get_output_element_type(0);
        auto indices_type = indices->get_output_element_type(0);
        switch (data_type.get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_gather_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_gather_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_gather<char>(data, indices, gather);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_gather<bfloat16>(data, indices, gather);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_gather<float16>(data, indices, gather);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_gather<float>(data, indices, gather);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_gather<double>(data, indices, gather);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_gather<int8_t>(data, indices, gather);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_gather<int16_t>(data, indices, gather);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_gather<int32_t>(data, indices, gather);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_gather<int64_t>(data, indices, gather);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_gather<uint8_t>(data, indices, gather);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_gather<uint16_t>(data, indices, gather);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_gather<uint32_t>(data, indices, gather);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_gather<uint64_t>(data, indices, gather);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto gather_matcher =
        make_shared<pattern::Matcher>(gather_op, "ConstantFolding.ConstantGather");
    this->add_matcher(gather_matcher, constant_gather_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

template <class T>
shared_ptr<op::Constant> fold_constant_slice(shared_ptr<op::Constant> constant,
                                             shared_ptr<op::Slice> slice)
{
    auto out_shape = slice->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::slice<T>(constant->get_data_ptr<T>(),
                                 out_vec.data(),
                                 constant->get_shape(),
                                 slice->get_lower_bounds(),
                                 slice->get_upper_bounds(),
                                 slice->get_strides(),
                                 out_shape);

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_slice()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto slice_op = make_shared<op::Slice>(
        data_label, Coordinate{1, 1, 1}, Coordinate{2, 3, 4}, Strides{1, 1, 2});

    auto constant_slice_callback = [data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_slice_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        auto slice = static_pointer_cast<op::Slice>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(slice));

        std::shared_ptr<op::Constant> replacement;

        switch (slice->get_output_element_type(0).get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_slice");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_slice");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_slice<char>(data_node, slice);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_slice<bfloat16>(data_node, slice);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_slice<float16>(data_node, slice);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_slice<float>(data_node, slice);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_slice<double>(data_node, slice);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_slice<int8_t>(data_node, slice);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_slice<int16_t>(data_node, slice);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_slice<int32_t>(data_node, slice);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_slice<int64_t>(data_node, slice);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_slice<uint8_t>(data_node, slice);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_slice<uint16_t>(data_node, slice);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_slice<uint32_t>(data_node, slice);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_slice<uint64_t>(data_node, slice);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto slice_matcher = make_shared<pattern::Matcher>(slice_op, "ConstantFolding.ConstantSlice");
    this->add_matcher(slice_matcher, constant_slice_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

template <class T>
shared_ptr<op::Constant> fold_constant_dyn_slice(shared_ptr<op::Constant> data,
                                                 shared_ptr<op::Constant> lb,
                                                 shared_ptr<op::Constant> ub,
                                                 shared_ptr<op::Constant> strides,
                                                 shared_ptr<op::DynSlice> slice)
{
    SlicePlan plan = make_slice_plan(data->get_shape(),
                                     lb->get_vector<int64_t>(),
                                     ub->get_vector<int64_t>(),
                                     strides->get_vector<int64_t>(),
                                     slice->get_lower_bounds_mask(),
                                     slice->get_upper_bounds_mask(),
                                     slice->get_new_axis(),
                                     slice->get_shrink_axis(),
                                     slice->get_ellipsis_mask());

    vector<T> slice_out_vec(shape_size(plan.reshape_in_shape));
    runtime::reference::slice<T>(data->get_data_ptr<T>(),
                                 slice_out_vec.data(),
                                 data->get_shape(),
                                 Coordinate(plan.begins.begin(), plan.begins.end()),
                                 Coordinate(plan.ends.begin(), plan.ends.end()),
                                 Strides(plan.strides.begin(), plan.strides.end()),
                                 plan.reshape_in_shape);

    vector<T> reshape_out_vec(shape_size(plan.reshape_out_shape));
    runtime::reference::reshape<T>(slice_out_vec.data(),
                                   reshape_out_vec.data(),
                                   plan.reshape_in_shape,
                                   get_default_order(plan.reshape_in_shape.size()),
                                   plan.reshape_out_shape);

    vector<T> reverse_out_vec(shape_size(plan.reshape_out_shape));
    runtime::reference::reverse<T>(reshape_out_vec.data(),
                                   reverse_out_vec.data(),
                                   plan.reshape_out_shape,
                                   plan.reshape_out_shape,
                                   plan.reverse_axes);

    return make_shared<op::Constant>(
        data->get_element_type(), plan.reshape_out_shape, reverse_out_vec);
}

void pass::ConstantFolding::construct_constant_dyn_slice()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto lb_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto ub_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto strides_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto dyn_slice_op = make_shared<op::DynSlice>(data_label,
                                                  lb_label,
                                                  ub_label,
                                                  strides_label,
                                                  AxisSet{},
                                                  AxisSet{},
                                                  AxisSet{},
                                                  AxisSet{},
                                                  AxisSet{});

    auto constant_dyn_slice_callback = [data_label, lb_label, ub_label, strides_label](
        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_dyn_slice_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        auto lb_node = static_pointer_cast<op::Constant>(pattern_map[lb_label]);
        auto ub_node = static_pointer_cast<op::Constant>(pattern_map[ub_label]);
        auto strides_node = static_pointer_cast<op::Constant>(pattern_map[strides_label]);
        auto dyn_slice = static_pointer_cast<op::DynSlice>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(dyn_slice));

        std::shared_ptr<op::Constant> replacement;

        switch (dyn_slice->get_output_element_type(0).get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_dyn_slice");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_dyn_slice");
            break;
        case element::Type_t::boolean:
            replacement =
                fold_constant_dyn_slice<char>(data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_dyn_slice<bfloat16>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_dyn_slice<float16>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_dyn_slice<float>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_dyn_slice<double>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_dyn_slice<int8_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_dyn_slice<int16_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_dyn_slice<int32_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_dyn_slice<int64_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_dyn_slice<uint8_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_dyn_slice<uint16_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_dyn_slice<uint32_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_dyn_slice<uint64_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto dyn_slice_matcher =
        make_shared<pattern::Matcher>(dyn_slice_op, "ConstantFolding.ConstantDynSlice");
    this->add_matcher(
        dyn_slice_matcher, constant_dyn_slice_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

template <class T>
shared_ptr<op::Constant> fold_constant_range(shared_ptr<op::Constant> start,
                                             shared_ptr<op::Constant> step,
                                             shared_ptr<op::Range> range)
{
    vector<T> out_vec(shape_size(range->get_shape()));
    runtime::reference::range<T>(start->get_vector<T>().data(),
                                 step->get_vector<T>().data(),
                                 range->get_shape(),
                                 out_vec.data());

    return make_shared<op::Constant>(range->get_element_type(), range->get_shape(), out_vec);
}

void pass::ConstantFolding::construct_constant_range()
{
    auto start_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto stop_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto step_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto range_op = make_shared<op::Range>(start_label, stop_label, step_label);

    auto constant_range_callback = [start_label, stop_label, step_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_range_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto start_node = static_pointer_cast<op::Constant>(pattern_map[start_label]);
        auto stop_node = static_pointer_cast<op::Constant>(pattern_map[stop_label]);
        auto step_node = static_pointer_cast<op::Constant>(pattern_map[step_label]);
        auto range = static_pointer_cast<op::Range>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(range));

        std::shared_ptr<op::Constant> replacement;

        switch (range->get_output_element_type(0).get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_range_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_range_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_range<char>(start_node, step_node, range);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_range<bfloat16>(start_node, step_node, range);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_range<float16>(start_node, step_node, range);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_range<float>(start_node, step_node, range);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_range<double>(start_node, step_node, range);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_range<int8_t>(start_node, step_node, range);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_range<int16_t>(start_node, step_node, range);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_range<int32_t>(start_node, step_node, range);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_range<int64_t>(start_node, step_node, range);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_range<uint8_t>(start_node, step_node, range);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_range<uint16_t>(start_node, step_node, range);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_range<uint32_t>(start_node, step_node, range);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_range<uint64_t>(start_node, step_node, range);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto range_matcher = make_shared<pattern::Matcher>(range_op, "ConstantFolding.ConstantRange");
    this->add_matcher(range_matcher, constant_range_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

template <class T>
shared_ptr<op::Constant> fold_constant_select(shared_ptr<op::Constant> selection,
                                              shared_ptr<op::Constant> t,
                                              shared_ptr<op::Constant> f,
                                              shared_ptr<op::Select> select)
{
    auto out_shape = select->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::select<T>(selection->get_data_ptr<char>(),
                                  t->get_data_ptr<T>(),
                                  f->get_data_ptr<T>(),
                                  out_vec.data(),
                                  shape_size(out_shape));

    return make_shared<op::Constant>(select->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_select()
{
    auto selection_label = make_shared<pattern::op::Label>(
        element::boolean, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto t_label = make_shared<pattern::op::Label>(
        element::i64, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto f_label = make_shared<pattern::op::Label>(
        element::i64, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto select_op = make_shared<op::Select>(selection_label, t_label, f_label);

    auto constant_select_callback = [selection_label, t_label, f_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_select_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto selection_node = static_pointer_cast<op::Constant>(pattern_map[selection_label]);
        auto t_node = static_pointer_cast<op::Constant>(pattern_map[t_label]);
        auto f_node = static_pointer_cast<op::Constant>(pattern_map[f_label]);
        auto select = static_pointer_cast<op::Select>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(select));

        std::shared_ptr<op::Constant> replacement;

        switch (select->get_output_element_type(0).get_type_enum())
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_select_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_select_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_select<char>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_select<bfloat16>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_select<float16>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_select<float>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_select<double>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_select<int8_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_select<int16_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_select<int32_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_select<int64_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_select<uint8_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_select<uint16_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_select<uint32_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_select<uint64_t>(selection_node, t_node, f_node, select);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto select_matcher =
        make_shared<pattern::Matcher>(select_op, "ConstantFolding.ConstantSelect");
    this->add_matcher(select_matcher, constant_select_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
