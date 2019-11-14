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

#include <cstdint>

#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/strided_slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/slice_plan.hpp"

#include <algorithm>

using namespace std;
using namespace ngraph;

#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/fused_op_tbl.hpp"
#include "ngraph/op/op_tbl.hpp"
};
#undef NGRAPH_OP

#define NGRAPH_OP(a, b) {#a, OP_TYPEID::a},
static unordered_map<string, OP_TYPEID> typeid_map{
#include "ngraph/op/fused_op_tbl.hpp"
#include "ngraph/op/op_tbl.hpp"
};
#undef NGRAPH_OP

static OP_TYPEID get_typeid(shared_ptr<Node> node)
{
    OP_TYPEID type_id;
    auto it = typeid_map.find(node->description());
    if (it != typeid_map.end())
    {
        type_id = it->second;
    }
    else
    {
        throw unsupported_op("Unsupported op '" + node->description() + "'");
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

    size_t op_version = node->get_version();

    if (op_version == 0)
    {
        return modified;
    }

    NGRAPH_CHECK(op_version == 1,
                 "Op version 1 transformation pass failed for ",
                 *node,
                 ", only op version 1 operations expected. Op version ",
                 op_version,
                 " found.");

// Not all enumeration values explicitly handled in switch
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (get_typeid(node))
    {
    case OP_TYPEID::Add:
    {
        downgrade_binary_elementwise_node<op::v0::Add, op::v1::Add>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::AvgPool:
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
    case OP_TYPEID::AvgPoolBackprop:
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
    case OP_TYPEID::Broadcast:
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
    case OP_TYPEID::Convolution:
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
    case OP_TYPEID::ConvolutionBackpropData:
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
    case OP_TYPEID::ConvolutionBackpropFilters:
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
    case OP_TYPEID::Divide:
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
    case OP_TYPEID::DynReshape:
    {
        auto tmp = as_type_ptr<op::v1::Reshape>(node);
        auto replacement_node = make_shared<op::v0::DynReshape>(node->input(0).get_source_output(),
                                                                node->input(1).get_source_output(),
                                                                tmp->get_zero_flag());
        replace_node(node, replacement_node);
        modified = true;
        break;
    }
    case OP_TYPEID::Equal:
    {
        downgrade_binary_elementwise_node<op::v0::Equal, op::v1::Equal>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::GenerateMask:
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
    case OP_TYPEID::Greater:
    {
        downgrade_binary_elementwise_node<op::v0::Greater, op::v1::Greater>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::GreaterEq:
    {
        downgrade_binary_elementwise_node<op::v0::GreaterEq, op::v1::GreaterEq>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Less:
    {
        downgrade_binary_elementwise_node<op::v0::Less, op::v1::Less>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::LessEqual:
    {
        downgrade_binary_elementwise_node<op::v0::LessEq, op::v1::LessEqual>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::LogicalAnd:
    {
        downgrade_binary_elementwise_node<op::v0::And, op::v1::LogicalAnd>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::LogicalNot:
    {
        replace_node(node, make_shared<op::v0::Not>(node->input(0).get_source_output()));
        modified = true;
        break;
    }
    case OP_TYPEID::LogicalOr:
    {
        downgrade_binary_elementwise_node<op::v0::Or, op::v1::LogicalOr>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::LogicalXor:
    {
        downgrade_binary_elementwise_node<op::v0::Xor, op::v1::LogicalXor>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Maximum:
    {
        downgrade_binary_elementwise_node<op::v0::Maximum, op::v1::Maximum>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::MaxPool:
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
    case OP_TYPEID::MaxPoolBackprop:
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
    case OP_TYPEID::Minimum:
    {
        downgrade_binary_elementwise_node<op::v0::Minimum, op::v1::Minimum>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Multiply:
    {
        downgrade_binary_elementwise_node<op::v0::Multiply, op::v1::Multiply>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::NotEqual:
    {
        downgrade_binary_elementwise_node<op::v0::NotEqual, op::v1::NotEqual>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Pad:
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
    case OP_TYPEID::Power:
    {
        downgrade_binary_elementwise_node<op::v0::Power, op::v1::Power>(node);
        modified = true;
        break;
    }
    case OP_TYPEID::Product:
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
    case OP_TYPEID::Reverse:
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
    case OP_TYPEID::Slice:
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
    case OP_TYPEID::Sum:
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
    case OP_TYPEID::TopK:
    {
        const auto tmp = as_type_ptr<op::v1::TopK>(node);
        const auto axis = tmp->get_axis();
        const auto sort_type = tmp->get_sort_type();
        const auto index_elem_type = tmp->get_index_element_type();

        bool comnpute_max;
        switch (tmp->get_mode())
        {
        case op::v1::TopK::Mode::MAX: comnpute_max = true; break;
        case op::v1::TopK::Mode::MIN: comnpute_max = false; break;
        default: break;
        }

        const auto arg_node = node->input_value(0);
        const auto k_node = node->input_value(1);

        auto replacement_node = make_shared<op::v0::TopK>(
            arg_node, k_node, axis, index_elem_type, comnpute_max, sort_type);

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
