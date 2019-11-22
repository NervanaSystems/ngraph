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
#include "ngraph/pass/opset1_upgrade.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/atan2.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/binary_convolution.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast_distributed.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/batch_mat_mul.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/dyn_pad.hpp"
#include "ngraph/op/experimental/dyn_replace_slice.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/layers/interpolate.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/experimental/random_uniform.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/experimental/tile.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/floor_mod.hpp"
#include "ngraph/op/fused/batch_mat_mul_transpose.hpp"
#include "ngraph/op/fused/clamp.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/crossentropy.hpp"
#include "ngraph/op/fused/depth_to_space.hpp"
#include "ngraph/op/fused/elu.hpp"
#include "ngraph/op/fused/fake_quantize.hpp"
#include "ngraph/op/fused/gelu.hpp"
#include "ngraph/op/fused/gemm.hpp"
#include "ngraph/op/fused/grn.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/fused/group_conv_transpose.hpp"
#include "ngraph/op/fused/gru_cell.hpp"
#include "ngraph/op/fused/hard_sigmoid.hpp"
#include "ngraph/op/fused/layer_norm.hpp"
#include "ngraph/op/fused/log_softmax.hpp"
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/op/fused/lstm_sequence.hpp"
#include "ngraph/op/fused/matmul.hpp"
#include "ngraph/op/fused/mvn.hpp"
#include "ngraph/op/fused/normalize_l2.hpp"
#include "ngraph/op/fused/partial_slice.hpp"
#include "ngraph/op/fused/prelu.hpp"
#include "ngraph/op/fused/reciprocal.hpp"
#include "ngraph/op/fused/rnn_cell.hpp"
#include "ngraph/op/fused/scale_shift.hpp"
#include "ngraph/op/fused/selu.hpp"
#include "ngraph/op/fused/shuffle_channels.hpp"
#include "ngraph/op/fused/softmax_crossentropy.hpp"
#include "ngraph/op/fused/space_to_depth.hpp"
#include "ngraph/op/fused/split.hpp"
#include "ngraph/op/fused/squared_difference.hpp"
#include "ngraph/op/fused/squeeze.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/gather_nd.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/recv.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/scatter_add.hpp"
#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/send.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/strided_slice.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/xor.hpp"

#include <limits>
#include <numeric>

using namespace std;
using namespace ngraph;

namespace
{
    enum class OP_TYPEID
    {
#define NGRAPH_OP(a, b) a,
#include "ngraph/op/op_v0_tbl.hpp"
#undef NGRAPH_OP
        OTHER
    };
}

static OP_TYPEID get_typeid(shared_ptr<Node> node)
{
    static map<NodeTypeInfo, OP_TYPEID> typeid_map{
#define NGRAPH_OP(a, b) {b::a::type_info, OP_TYPEID::a},
#include "ngraph/op/op_v0_tbl.hpp"
#undef NGRAPH_OP
    };
    OP_TYPEID type_id = OP_TYPEID::OTHER;
    auto it = typeid_map.find(node->get_type_info());
    if (it != typeid_map.end())
    {
        type_id = it->second;
    }
    return type_id;
}
// END mapping to OP_TYPEID

template <typename OpV0, typename OpV1>
void upgrade_binary_elementwise_node(const shared_ptr<Node>& node)
{
    const auto tmp = dynamic_cast<const OpV0*>(node.get());
    const auto autob = tmp->get_autob();
    auto replacement_node = make_shared<OpV1>(
        node->input(0).get_source_output(), node->input(1).get_source_output(), autob);
    replace_node(node, replacement_node);
}

bool pass::Opset1Upgrade::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;

// Not all enumeration values explicitly handled in switch
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (get_typeid(node))
    {
    case OP_TYPEID::Add:
    {
        upgrade_binary_elementwise_node<op::v0::Add, op::v1::Add>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::And:
    {
        upgrade_binary_elementwise_node<op::v0::And, op::v1::LogicalAnd>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::AvgPool:
    {
        auto tmp = dynamic_cast<const op::v0::AvgPool*>(node.get());

        auto rounding_type = static_cast<op::RoundingType>(tmp->get_ceil_mode());
        auto exclude_pad = !tmp->get_include_padding_in_avg_computation();
        auto auto_pad = tmp->get_pad_type();
        auto pads_begin = tmp->get_padding_below();
        auto pads_end = tmp->get_padding_above();
        auto strides = tmp->get_window_movement_strides();
        auto kernel = tmp->get_window_shape();

        auto replacement_node = make_shared<op::v1::AvgPool>(node->input(0).get_source_output(),
                                                             strides,
                                                             pads_begin,
                                                             pads_end,
                                                             kernel,
                                                             exclude_pad,
                                                             rounding_type,
                                                             auto_pad);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::AvgPoolBackprop:
    {
        auto tmp = dynamic_cast<const op::v0::AvgPoolBackprop*>(node.get());

        auto exclude_pad = !tmp->get_include_padding_in_avg_computation();
        auto pads_begin = tmp->get_padding_below();
        auto pads_end = tmp->get_padding_above();
        auto strides = tmp->get_window_movement_strides();
        auto kernel = tmp->get_window_shape();

        auto replacement_node =
            make_shared<op::v1::AvgPoolBackprop>(node->input(0).get_source_output(),
                                                 node->input(1).get_source_output(),
                                                 strides,
                                                 pads_begin,
                                                 pads_end,
                                                 kernel,
                                                 exclude_pad);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Broadcast:
    {
        auto tmp = dynamic_cast<const op::v0::Broadcast*>(node.get());
        auto result_shape = tmp->get_broadcast_shape();
        auto result_shape_node =
            op::Constant::create(element::i64, Shape{result_shape.size()}, result_shape);
        auto broadcast_axes = tmp->get_broadcast_axes();

        // Flip broadcast_axes to get axes_mapping
        std::vector<size_t> axes_mapping(result_shape.size());
        std::iota(axes_mapping.begin(), axes_mapping.end(), 0);
        for (auto i = broadcast_axes.rbegin(); i != broadcast_axes.rend(); i++)
        {
            axes_mapping.erase(axes_mapping.begin() + *i);
        }
        auto axes_mapping_node =
            op::Constant::create(element::i64, Shape{axes_mapping.size()}, axes_mapping);

        auto replacement_node = make_shared<op::v1::Broadcast>(node->input(0).get_source_output(),
                                                               result_shape_node->output(0),
                                                               axes_mapping_node->output(0));
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Convolution:
    {
        auto tmp = dynamic_cast<const op::v0::Convolution*>(node.get());
        auto strides = tmp->get_window_movement_strides();
        auto dilations = tmp->get_window_dilation_strides();
        auto pads_begin = tmp->get_padding_below();
        auto pads_end = tmp->get_padding_above();
        auto data_dilation_strides = tmp->get_data_dilation_strides();
        auto auto_pad = tmp->get_pad_type();

        bool is_dds_valid = true;
        for (auto value : data_dilation_strides)
        {
            is_dds_valid = is_dds_valid && (value == 1);
        }

        NGRAPH_CHECK(is_dds_valid,
                     "Unable to convert Convolution:0 to Convolution:1 with data dilation strides "
                     "other than `1`. Node: ",
                     *node);

        auto replacement_node = make_shared<op::v1::Convolution>(node->input(0).get_source_output(),
                                                                 node->input(1).get_source_output(),
                                                                 strides,
                                                                 pads_begin,
                                                                 pads_end,
                                                                 dilations,
                                                                 auto_pad);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::ConvolutionBackpropData:
    {
        auto tmp = dynamic_cast<const op::v0::ConvolutionBackpropData*>(node.get());
        auto data_batch_shape = tmp->get_data_batch_shape();
        auto strides = tmp->get_window_movement_strides_forward();
        auto dilations = tmp->get_window_dilation_strides_forward();
        auto pads_begin = tmp->get_padding_below_forward();
        auto pads_end = tmp->get_padding_above_forward();
        auto data_dilation_strides = tmp->get_data_dilation_strides_forward();

        bool is_dds_valid = true;
        for (auto value : data_dilation_strides)
        {
            is_dds_valid = is_dds_valid && (value == 1);
        }

        NGRAPH_CHECK(is_dds_valid,
                     "Unable to convert ConvolutionBackpropData:0 to ConvolutionBackpropData:1 "
                     "with data dilation strides "
                     "other than `1`. Node: ",
                     *node);

        auto replacement_node =
            make_shared<op::v1::ConvolutionBackpropData>(node->input(0).get_source_output(),
                                                         node->input(1).get_source_output(),
                                                         node->input(2).get_source_output(),
                                                         strides,
                                                         dilations,
                                                         pads_begin,
                                                         pads_end);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::ConvolutionBackpropFilters:
    {
        auto tmp = dynamic_cast<const op::v0::ConvolutionBackpropFilters*>(node.get());
        auto filters_shape = tmp->get_filters_shape();
        auto strides = tmp->get_window_movement_strides_forward();
        auto dilations = tmp->get_window_dilation_strides_forward();
        auto pads_begin = tmp->get_padding_below_forward();
        auto pads_end = tmp->get_padding_above_forward();
        auto data_dilation_strides = tmp->get_data_dilation_strides_forward();

        bool is_dds_valid = true;
        for (auto value : data_dilation_strides)
        {
            is_dds_valid = is_dds_valid && (value == 1);
        }

        NGRAPH_CHECK(
            is_dds_valid,
            "Unable to convert ConvolutionBackpropFilters:0 to ConvolutionBackpropFilters:1 "
            "with data dilation strides "
            "other than `1`. Node: ",
            *node);

        auto replacement_node =
            make_shared<op::v1::ConvolutionBackpropFilters>(node->input(0).get_source_output(),
                                                            node->input(1).get_source_output(),
                                                            node->input(2).get_source_output(),
                                                            strides,
                                                            dilations,
                                                            pads_begin,
                                                            pads_end);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Divide:
    {
        const auto tmp = dynamic_cast<const op::v0::Divide*>(node.get());
        const auto autob = tmp->get_autob();
        const bool pydiv = tmp->is_pythondiv();
        auto replacement_node = make_shared<op::v1::Divide>(
            node->input(0).get_source_output(), node->input(1).get_source_output(), pydiv, autob);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::DynReshape:
    {
        auto zero_flag = false;
        auto replacement_node = make_shared<op::v1::Reshape>(
            node->input(0).get_source_output(), node->input(1).get_source_output(), zero_flag);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Equal:
    {
        upgrade_binary_elementwise_node<op::v0::Equal, op::v1::Equal>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Gather:
    {
        auto tmp = dynamic_cast<const op::v0::Gather*>(node.get());
        int64_t axis = tmp->get_axis();

        auto axis_node = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{axis});
        auto replacement_node = make_shared<op::v1::Gather>(
            node->input(0).get_source_output(), node->input(1).get_source_output(), axis_node);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Greater:
    {
        upgrade_binary_elementwise_node<op::v0::Greater, op::v1::Greater>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::GreaterEq:
    {
        upgrade_binary_elementwise_node<op::v0::GreaterEq, op::v1::GreaterEq>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Less:
    {
        upgrade_binary_elementwise_node<op::v0::Less, op::v1::Less>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::LessEq:
    {
        upgrade_binary_elementwise_node<op::v0::LessEq, op::v1::LessEqual>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Maximum:
    {
        upgrade_binary_elementwise_node<op::v0::Maximum, op::v1::Maximum>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::MaxPool:
    {
        auto tmp = dynamic_cast<const op::v0::MaxPool*>(node.get());

        auto rounding_type = static_cast<op::RoundingType>(tmp->get_ceil_mode());
        auto auto_pad = tmp->get_pad_type();
        auto pads_begin = tmp->get_padding_below();
        auto pads_end = tmp->get_padding_above();
        auto strides = tmp->get_window_movement_strides();
        auto kernel = tmp->get_window_shape();

        auto replacement_node = make_shared<op::v1::MaxPool>(node->input(0).get_source_output(),
                                                             strides,
                                                             pads_begin,
                                                             pads_end,
                                                             kernel,
                                                             rounding_type,
                                                             auto_pad);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::MaxPoolBackprop:
    {
        auto tmp = dynamic_cast<const op::v0::MaxPoolBackprop*>(node.get());

        auto pads_begin = tmp->get_padding_below();
        auto pads_end = tmp->get_padding_above();
        auto strides = tmp->get_window_movement_strides();
        auto kernel = tmp->get_window_shape();

        shared_ptr<Node> replacement_node;
        if (node->get_inputs().size() == 3)
        {
            replacement_node =
                make_shared<op::v1::MaxPoolBackprop>(node->input(0).get_source_output(),
                                                     node->input(1).get_source_output(),
                                                     node->input(2).get_source_output(),
                                                     strides,
                                                     pads_begin,
                                                     pads_end,
                                                     kernel);
        }
        else
        {
            replacement_node =
                make_shared<op::v1::MaxPoolBackprop>(node->input(0).get_source_output(),
                                                     node->input(1).get_source_output(),
                                                     strides,
                                                     pads_begin,
                                                     pads_end,
                                                     kernel);
        }
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Minimum:
    {
        upgrade_binary_elementwise_node<op::v0::Minimum, op::v1::Minimum>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Multiply:
    {
        upgrade_binary_elementwise_node<op::v0::Multiply, op::v1::Multiply>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Not:
    {
        replace_node(node, make_shared<op::v1::LogicalNot>(node->input(0).get_source_output()));
        modified = true;
        break;
    }
    case OP_TYPEID::NotEqual:
    {
        upgrade_binary_elementwise_node<op::v0::NotEqual, op::v1::NotEqual>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Or:
    {
        upgrade_binary_elementwise_node<op::v0::Or, op::v1::LogicalOr>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Pad:
    {
        auto tmp = dynamic_cast<const op::v0::Pad*>(node.get());
        auto padding_below = tmp->get_padding_below();
        auto pads_begin_node =
            make_shared<op::Constant>(element::i64, Shape{padding_below.size()}, padding_below);
        auto padding_above = tmp->get_padding_above();
        auto pads_end_node =
            make_shared<op::Constant>(element::i64, Shape{padding_above.size()}, padding_above);

        auto replacement_node = make_shared<op::v1::Pad>(node->input(0).get_source_output(),
                                                         pads_begin_node,
                                                         pads_end_node,
                                                         node->input(1).get_source_output(),
                                                         tmp->get_pad_mode());

        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Power:
    {
        upgrade_binary_elementwise_node<op::v0::Power, op::v1::Power>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Product:
    {
        bool keep_dims = false;
        auto replacement_node = make_shared<op::v1::ReduceProd>(
            node->input(0).get_source_output(), node->input(1).get_source_output(), keep_dims);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Reverse:
    {
        // creates a Constant node from the v0::Reverse reversed_axes attribute
        // and uses it as the second input of v1::Reverse
        const auto reverse_v0 = dynamic_cast<const op::Reverse*>(node.get());
        const auto reversed_axes = reverse_v0->get_reversed_axes();

        const auto reversed_axes_constant = op::Constant::create(
            element::i64, Shape{reversed_axes.size()}, reversed_axes.to_vector());

        const auto reverse_v1 = make_shared<op::v1::Reverse>(node->input(0).get_source_output(),
                                                             reversed_axes_constant,
                                                             op::v1::Reverse::Mode::INDEX);

        replace_node(node, reverse_v1);
        modified = true;

        break;
    }
    case OP_TYPEID::Softmax:
    {
        auto tmp = dynamic_cast<const op::v0::Softmax*>(node.get());

        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant(),
                     "axes parameter is expected to be a static constant");

        AxisSet axes = tmp->get_axes();

        NGRAPH_CHECK(
            axes.size() == 1,
            "Unable to convert Softmax:0 to Softmax:1 with zero or more than one axis. Node: ",
            *node);

        auto replacement_node =
            make_shared<op::v1::Softmax>(node->input(0).get_source_output(), axes.to_vector()[0]);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Slice:
    {
        const auto tmp = as_type_ptr<op::v0::Slice>(node);

        const auto data = node->input(0).get_source_output();
        const auto begin = op::Constant::create(
            element::i64, Shape{tmp->get_lower_bounds().size()}, tmp->get_lower_bounds());
        const auto end = op::Constant::create(
            element::i64, Shape{tmp->get_upper_bounds().size()}, tmp->get_upper_bounds());
        const auto strides = op::Constant::create(
            element::i64, Shape{tmp->get_strides().size()}, tmp->get_strides());
        int64_t input_size = tmp->get_lower_bounds().size();

        auto replacement_node = make_shared<op::v1::StridedSlice>(data,
                                                                  begin,
                                                                  end,
                                                                  strides,
                                                                  vector<int64_t>(input_size, 0),
                                                                  vector<int64_t>(input_size, 0));

        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Sum:
    {
        bool keep_dims = false;
        auto replacement_node = make_shared<op::v1::ReduceSum>(
            node->input(0).get_source_output(), node->input(1).get_source_output(), keep_dims);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::TopK:
    {
        const auto topk_v0 = dynamic_cast<const op::TopK*>(node.get());

        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant(),
                     "parameter k is expected to be a static constant");
        NGRAPH_CHECK(node->input_value(2).get_node_shared_ptr()->is_constant(),
                     "parameter top_k_axis is expected to be a static constant");

        const auto k = topk_v0->get_k();
        const auto axis = topk_v0->get_top_k_axis();

        std::string sort;
        switch (topk_v0->get_sort())
        {
        case op::TopK::SortType::SORT_INDICES: sort = "index"; break;
        case op::TopK::SortType::SORT_VALUES: sort = "value"; break;
        default: sort = "none"; break;
        }

        std::string mode;
        if (topk_v0->get_compute_max())
        {
            mode = "max";
        }
        else
        {
            mode = "min";
        }

        const auto k_constant = op::Constant::create(element::i64, Shape{}, {k});
        auto replacement_node =
            make_shared<op::v1::TopK>(node->input_value(0), k_constant, axis, mode, sort);

        // indices output will be 0, values 1
        vector<int64_t> output_order{1, 0};
        replace_node(node, replacement_node, output_order);
        modified = true;
        break;
    }
    case OP_TYPEID::Xor:
    {
        const auto xor_v0 = dynamic_cast<const op::v0::Xor*>(node.get());
        auto replacement_node = make_shared<op::v1::LogicalXor>(node->input(0).get_source_output(),
                                                                node->input(1).get_source_output(),
                                                                xor_v0->get_autob());
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    default: break;
    }

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    return modified;
}
