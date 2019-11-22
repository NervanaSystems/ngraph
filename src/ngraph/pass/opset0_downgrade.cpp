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

#include <algorithm>
#include <cstdint>
#include <numeric>

#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
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
#include "ngraph/op/variadic_split.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/slice_plan.hpp"
#include "ngraph/type.hpp"

using namespace std;
using namespace ngraph;

namespace
{
    enum class OP_TYPEID
    {
#define NGRAPH_OP(a, b) a,
//#include "ngraph/op/fused_op_tbl.hpp"
//#include "ngraph/op/op_v0_tbl.hpp"
#undef NGRAPH_OP
#define NGRAPH_OP(a, b) a##_v1,
#include "ngraph/op/op_v1_tbl.hpp"
        OTHER
    };
#undef NGRAPH_OP
}

static OP_TYPEID get_typeid(shared_ptr<Node> node)
{
    static map<NodeTypeInfo, OP_TYPEID> typeid_map{
#define NGRAPH_OP(a, b) {b::a::type_info, OP_TYPEID::a##_v1},
#include "ngraph/op/op_v1_tbl.hpp"
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
void downgrade_binary_elementwise_node(const shared_ptr<Node>& node)
{
    const auto tmp = as_type_ptr<OpV1>(node);
    const auto input_arg0 = node->input(0).get_source_output();
    const auto input_arg1 = node->input(1).get_source_output();
    const auto autob = tmp->get_autob();
    auto replacement_node = make_shared<OpV0>(input_arg0, input_arg1, autob);
    replace_node(node, replacement_node);
}

bool pass::Opset0Downgrade::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;

// Not all enumeration values explicitly handled in switch
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (get_typeid(node))
    {
    case OP_TYPEID::Add_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Add, op::v1::Add>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::AvgPool_v1:
    {
        const auto tmp = as_type_ptr<op::v1::AvgPool>(node);

        auto const input_arg = node->input(0).get_source_output();
        const auto ceil_mode = static_cast<bool>(tmp->get_rounding_type());
        const auto include_padding_in_avg_computation = !tmp->get_exclude_pad();
        const auto pad_type = tmp->get_auto_pad();
        const auto padding_below = tmp->get_pads_begin();
        const auto padding_above = tmp->get_pads_end();
        const auto window_movement_strides = tmp->get_strides();
        const auto window_shape = tmp->get_kernel();

        auto replacement_node = make_shared<op::v0::AvgPool>(input_arg,
                                                             window_shape,
                                                             window_movement_strides,
                                                             padding_below,
                                                             padding_above,
                                                             include_padding_in_avg_computation,
                                                             pad_type,
                                                             ceil_mode);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::AvgPoolBackprop_v1:
    {
        const auto tmp = as_type_ptr<op::v1::AvgPoolBackprop>(node);
        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant());
        const auto forward_arg_shape =
            static_pointer_cast<op::Constant>(node->input_value(1).get_node_shared_ptr())
                ->get_shape_val();
        const auto delta = node->input(0).get_source_output();
        const auto include_padding_in_avg_computation = !tmp->get_exclude_pad();
        const auto padding_below = tmp->get_pads_begin();
        const auto padding_above = tmp->get_pads_end();
        const auto window_movement_strides = tmp->get_strides();
        const auto window_shape = tmp->get_kernel();

        auto replacement_node =
            make_shared<op::v0::AvgPoolBackprop>(forward_arg_shape,
                                                 delta,
                                                 window_shape,
                                                 window_movement_strides,
                                                 padding_below,
                                                 padding_above,
                                                 include_padding_in_avg_computation);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Broadcast_v1:
    {
        auto tmp = dynamic_cast<const op::v1::Broadcast*>(node.get());
        const auto arg = node->input(0).get_source_output();
        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant());
        auto target_shape =
            static_pointer_cast<op::Constant>(node->input_value(1).get_node_shared_ptr())
                ->get_shape_val();
        NGRAPH_CHECK(tmp->get_broadcast_axes().first);
        auto replacement_node =
            make_shared<op::v0::Broadcast>(arg, target_shape, tmp->get_broadcast_axes().second);

        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Convolution_v1:
    {
        auto tmp = as_type_ptr<op::v1::Convolution>(node);
        const auto data_arg = node->input(0).get_source_output();
        const auto filters_arg = node->input(1).get_source_output();
        const PartialShape& data_arg_pshape = node->get_input_partial_shape(0);
        NGRAPH_CHECK(data_arg_pshape.rank().is_static(),
                     "Unable to convert Convolution:v1 to Convolution:v0 if data argument "
                     "rank is dynamic. Node: ",
                     *node);
        const size_t num_spatial_dims = static_cast<size_t>(data_arg_pshape.rank()) - 2;
        auto replacement_node = make_shared<op::v0::Convolution>(data_arg,
                                                                 filters_arg,
                                                                 tmp->get_strides(),
                                                                 tmp->get_dilations(),
                                                                 tmp->get_pads_begin(),
                                                                 tmp->get_pads_end(),
                                                                 Strides(num_spatial_dims, 1),
                                                                 tmp->get_auto_pad());
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::ConvolutionBackpropData_v1:
    {
        auto tmp = as_type_ptr<op::v1::ConvolutionBackpropData>(node);
        NGRAPH_CHECK(node->input_value(2).get_node_shared_ptr()->is_constant());
        auto data_batch_shape =
            static_pointer_cast<op::Constant>(node->input_value(2).get_node_shared_ptr())
                ->get_shape_val();
        const auto filters_arg = node->input(0).get_source_output();
        const auto delta_arg = node->input(1).get_source_output();
        const PartialShape& delta_arg_pshape = node->get_input_partial_shape(1);
        NGRAPH_CHECK(delta_arg_pshape.rank().is_static(),
                     "Unable to convert ConvolutionBackpropData:v1 to ConvolutionBackpropData:v0 "
                     "if delta argument rank is dynamic. Node: ",
                     *node);
        const size_t num_spatial_dims = static_cast<size_t>(delta_arg_pshape.rank()) - 2;
        auto replacement_node =
            make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                         filters_arg,
                                                         delta_arg,
                                                         tmp->get_strides(),
                                                         tmp->get_dilations(),
                                                         tmp->get_pads_begin(),
                                                         tmp->get_pads_end(),
                                                         Strides(num_spatial_dims, 1));
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::ConvolutionBackpropFilters_v1:
    {
        auto tmp = as_type_ptr<op::v1::ConvolutionBackpropFilters>(node);
        NGRAPH_CHECK(node->input_value(2).get_node_shared_ptr()->is_constant());
        auto filters_shape =
            static_pointer_cast<op::Constant>(node->input_value(2).get_node_shared_ptr())
                ->get_shape_val();
        const auto data_arg = node->input(0).get_source_output();
        const auto delta_arg = node->input(1).get_source_output();
        const PartialShape& data_arg_pshape = node->get_input_partial_shape(0);
        NGRAPH_CHECK(data_arg_pshape.rank().is_static(),
                     "Unable to convert ConvolutionBackpropFilters:v1 to "
                     "ConvolutionBackpropFilters:v0 if data argument rank is dynamic. Node: ",
                     *node);
        const size_t num_spatial_dims = static_cast<size_t>(data_arg_pshape.rank()) - 2;
        auto replacement_node =
            make_shared<op::v0::ConvolutionBackpropFilters>(data_arg,
                                                            filters_shape,
                                                            delta_arg,
                                                            tmp->get_strides(),
                                                            tmp->get_dilations(),
                                                            tmp->get_pads_begin(),
                                                            tmp->get_pads_end(),
                                                            Strides(num_spatial_dims, 1));
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Divide_v1:
    {
        const auto tmp = as_type_ptr<op::v1::Divide>(node);
        const auto input_arg0 = node->input(0).get_source_output();
        const auto input_arg1 = node->input(1).get_source_output();
        const auto autob = tmp->get_autob();
        const bool pydiv = tmp->is_pythondiv();
        auto replacement_node = make_shared<op::v0::Divide>(input_arg0, input_arg1, pydiv, autob);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Reshape_v1:
    {
        auto tmp = as_type_ptr<op::v1::Reshape>(node);
        auto replacement_node = make_shared<op::v0::DynReshape>(node->input(0).get_source_output(),
                                                                node->input(1).get_source_output(),
                                                                tmp->get_zero_flag());
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Equal_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Equal, op::v1::Equal>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::GenerateMask_v1:
    {
        auto tmp = dynamic_cast<const op::v1::GenerateMask*>(node.get());
        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant());
        auto mask_shape =
            static_pointer_cast<op::Constant>(node->input_value(1).get_node_shared_ptr())
                ->get_shape_val();
        auto seed = tmp->get_seed();
        auto use_seed = tmp->get_use_seed();
        auto probability = tmp->get_probability();
        auto et = tmp->get_element_type();

        auto replacement_node = make_shared<op::v0::GenerateMask>(
            node->input(0).get_source_output(), mask_shape, et, seed, probability, use_seed);

        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Greater_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Greater, op::v1::Greater>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::GreaterEq_v1:
    {
        downgrade_binary_elementwise_node<op::v0::GreaterEq, op::v1::GreaterEq>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Less_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Less, op::v1::Less>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::LessEqual_v1:
    {
        downgrade_binary_elementwise_node<op::v0::LessEq, op::v1::LessEqual>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::LogicalAnd_v1:
    {
        downgrade_binary_elementwise_node<op::v0::And, op::v1::LogicalAnd>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::LogicalNot_v1:
    {
        replace_node(node, make_shared<op::v0::Not>(node->input(0).get_source_output()));
        modified = true;
        break;
    }
    case OP_TYPEID::LogicalOr_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Or, op::v1::LogicalOr>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::LogicalXor_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Xor, op::v1::LogicalXor>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Maximum_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Maximum, op::v1::Maximum>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::MaxPool_v1:
    {
        auto tmp = as_type_ptr<op::v1::MaxPool>(node);

        auto const input_arg = node->input(0).get_source_output();
        auto ceil_mode = static_cast<bool>(tmp->get_rounding_type());
        auto pad_type = tmp->get_auto_pad();
        auto padding_below = tmp->get_pads_begin();
        auto padding_above = tmp->get_pads_end();
        auto window_movement_strides = tmp->get_strides();
        auto window_shape = tmp->get_kernel();

        auto replacement_node = make_shared<op::v0::MaxPool>(input_arg,
                                                             window_shape,
                                                             window_movement_strides,
                                                             padding_below,
                                                             padding_above,
                                                             pad_type,
                                                             ceil_mode);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::MaxPoolBackprop_v1:
    {
        const auto tmp = as_type_ptr<op::v1::MaxPoolBackprop>(node);

        const auto padding_below = tmp->get_pads_begin();
        const auto padding_above = tmp->get_pads_end();
        const auto window_movement_strides = tmp->get_strides();
        const auto window_shape = tmp->get_kernel();

        const auto arg_forward = node->input(0).get_source_output();
        const auto delta = node->input(1).get_source_output();

        shared_ptr<Node> replacement_node;
        if (node->get_inputs().size() == 3)
        {
            const auto result_forward = node->input(2).get_source_output();
            replacement_node = make_shared<op::v0::MaxPoolBackprop>(arg_forward,
                                                                    delta,
                                                                    result_forward,
                                                                    window_shape,
                                                                    window_movement_strides,
                                                                    padding_below,
                                                                    padding_above);
        }
        else
        {
            replacement_node = make_shared<op::v0::MaxPoolBackprop>(arg_forward,
                                                                    delta,
                                                                    window_movement_strides,
                                                                    window_shape,
                                                                    padding_below,
                                                                    padding_above);
        }
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Minimum_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Minimum, op::v1::Minimum>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Multiply_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Multiply, op::v1::Multiply>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::NotEqual_v1:
    {
        downgrade_binary_elementwise_node<op::v0::NotEqual, op::v1::NotEqual>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Pad_v1:
    {
        auto tmp = as_type_ptr<op::v1::Pad>(node);
        const auto pad_arg = node->input(0).get_source_output();
        const auto pad_value = node->input(3).get_source_output();
        auto replacement_node = make_shared<op::v0::Pad>(
            pad_arg, pad_value, tmp->get_pads_begin(), tmp->get_pads_end(), tmp->get_pad_mode());

        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Power_v1:
    {
        downgrade_binary_elementwise_node<op::v0::Power, op::v1::Power>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::ReduceProd_v1:
    {
        auto tmp = as_type_ptr<op::v1::ReduceProd>(node);
        auto replacement_node = make_shared<op::v0::Product>(node->input(0).get_source_output(),
                                                             node->input(1).get_source_output());
        if (tmp->get_keep_dims())
        {
            NGRAPH_CHECK(tmp->reduction_axes_constant(),
                         "Unable to convert ReduceProd:v1 to Product:v0 "
                         "if reduction axes are not constant (for keep_dims=true). Node: ",
                         *node);
            auto output_pshape = replacement_node->get_output_partial_shape(0);
            NGRAPH_CHECK(output_pshape.is_static(),
                         "Unable to convert ReduceProd:v1 to Product:v0 "
                         "if output shape is dynamic (for keep_dims=true). Node: ",
                         *node);
            const auto output_shape = output_pshape.to_shape();
            auto reshaped_output_shape = output_shape;
            for (const auto& axis : tmp->get_reduction_axes())
            {
                reshaped_output_shape.insert(reshaped_output_shape.begin() + axis, 1);
            }
            auto reshaped_product = make_shared<op::Reshape>(replacement_node->output(0),
                                                             get_default_order(output_shape),
                                                             reshaped_output_shape);
            replace_node(node, reshaped_product);
        }
        else
        {
            replace_node(node, replacement_node);
        }
        modified = true;
        break;
    }
    case OP_TYPEID::Reverse_v1:
    {
        auto tmp = as_type_ptr<op::v1::Reverse>(node);
        auto axes_node = tmp->input_value(1).get_node_shared_ptr();
        NGRAPH_CHECK(axes_node->is_constant(),
                     "Unable to convert Reverse:v1 to Reverse:v0 "
                     "if reduction axes are not constant. Node: ",
                     *node);
        const auto axes_node_const = as_type_ptr<op::Constant>(axes_node);
        AxisSet axes{};
        if (tmp->get_mode() == op::v1::Reverse::Mode::INDEX)
        {
            axes = axes_node_const->get_axis_vector_val();
        }
        else // Mode::MASK
        {
            auto axes_mask = axes_node_const->get_vector<bool>();
            for (size_t i = 0; i < axes_mask.size(); ++i)
            {
                if (axes_mask[i])
                {
                    axes.emplace(i);
                }
            }
        }
        auto replacement_node =
            make_shared<op::v0::Reverse>(node->input(0).get_source_output(), axes);

        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::StridedSlice_v1:
    {
        auto convert_mask_to_axes = [](const std::vector<int64_t>& mask) {
            AxisSet axes{};
            for (auto i = 0; i < mask.size(); ++i)
            {
                if (mask[i] == 1)
                {
                    axes.emplace(i);
                }
            }
            return axes;
        };

        const auto input_data = node->input_value(0);
        const auto input_data_pshape = input_data.get_partial_shape();

        NGRAPH_CHECK(input_data_pshape.is_static(),
                     "Unable to convert StridedSlice:v1 to Slice:v0 "
                     "if input rank is not static. Node: ",
                     *node);

        const auto begin_const =
            as_type_ptr<op::Constant>(node->input_value(1).get_node_shared_ptr());
        const auto end_const =
            as_type_ptr<op::Constant>(node->input_value(2).get_node_shared_ptr());
        const auto strides = as_type_ptr<op::Constant>(node->input_value(3).get_node_shared_ptr());

        NGRAPH_CHECK(begin_const && end_const && strides,
                     "Unable to convert StridedSlice:v1 to Slice:v0 "
                     "if begin, end or strides are not constant. Node: ",
                     *node);

        const auto tmp = as_type_ptr<op::v1::StridedSlice>(node);

        SlicePlan p = make_slice_plan(input_data_pshape.to_shape(),
                                      begin_const->get_vector<int64_t>(),
                                      end_const->get_vector<int64_t>(),
                                      strides->get_vector<int64_t>(),
                                      convert_mask_to_axes(tmp->get_begin_mask()),
                                      convert_mask_to_axes(tmp->get_end_mask()),
                                      convert_mask_to_axes(tmp->get_new_axis_mask()),
                                      convert_mask_to_axes(tmp->get_shrink_axis_mask()),
                                      convert_mask_to_axes(tmp->get_ellipsis_mask()));

        shared_ptr<Node> replacement_node =
            make_shared<op::v0::Slice>(input_data,
                                       Coordinate(p.begins.begin(), p.begins.end()),
                                       Coordinate(p.ends.begin(), p.ends.end()),
                                       Strides(p.strides.begin(), p.strides.end()));

        if (p.reshape_in_shape != p.reshape_out_shape)
        {
            replacement_node =
                make_shared<op::Reshape>(replacement_node,
                                         ngraph::get_default_order(p.reshape_in_shape),
                                         p.reshape_out_shape);
        }

        if (!p.reverse_axes.empty())
        {
            replacement_node = make_shared<op::Reverse>(replacement_node, p.reverse_axes);
        }

        replace_node(node, replacement_node);
        break;
    }
    case OP_TYPEID::Softmax_v1:
    {
        auto tmp = as_type_ptr<op::v1::Softmax>(node);
        auto axis = tmp->get_axis();
        auto data = node->input(0);
        auto data_shape = data.get_shape();
        std::vector<size_t> axes(data_shape.size() - axis);
        std::iota(std::begin(axes), std::end(axes), axis);
        auto replacement_node =
            make_shared<op::v0::Softmax>(node->input(0).get_source_output(), axes);
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::ReduceSum_v1:
    {
        auto tmp = as_type_ptr<op::v1::ReduceSum>(node);
        auto replacement_node = make_shared<op::v0::Sum>(node->input(0).get_source_output(),
                                                         node->input(1).get_source_output());
        if (tmp->get_keep_dims())
        {
            NGRAPH_CHECK(tmp->reduction_axes_constant(),
                         "Unable to convert ReduceSum:v1 to Sum:v0 "
                         "if reduction axes are not constant (for keep_dims=true). Node: ",
                         *node);
            auto output_pshape = replacement_node->get_output_partial_shape(0);
            NGRAPH_CHECK(output_pshape.is_static(),
                         "Unable to convert ReduceSum:v1 to Sum:v0 "
                         "if output shape is dynamic (for keep_dims=true). Node: ",
                         *node);
            const auto output_shape = output_pshape.to_shape();
            auto reshaped_output_shape = output_shape;
            for (const auto& axis : tmp->get_reduction_axes())
            {
                reshaped_output_shape.insert(reshaped_output_shape.begin() + axis, 1);
            }
            auto reshaped_product = make_shared<op::Reshape>(replacement_node->output(0),
                                                             get_default_order(output_shape),
                                                             reshaped_output_shape);
            replace_node(node, reshaped_product);
        }
        else
        {
            replace_node(node, replacement_node);
        }
        modified = true;
        break;
    }
    case OP_TYPEID::TopK_v1:
    {
        const auto tmp = as_type_ptr<op::v1::TopK>(node);
        const auto axis = tmp->get_axis();
        const auto sort_type = tmp->get_sort_type();
        const auto index_elem_type = tmp->get_index_element_type();

        bool compute_max;
        switch (tmp->get_mode())
        {
        case op::v1::TopK::Mode::MAX: compute_max = true; break;
        case op::v1::TopK::Mode::MIN: compute_max = false; break;
        default: break;
        }

        const auto arg_node = node->input_value(0);
        const auto k_node = node->input_value(1);

        auto replacement_node = make_shared<op::v0::TopK>(
            arg_node, k_node, axis, index_elem_type, compute_max, sort_type);

        // values output will be 0, indices 1
        vector<int64_t> output_order{1, 0};
        replace_node(node, replacement_node, output_order);
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
