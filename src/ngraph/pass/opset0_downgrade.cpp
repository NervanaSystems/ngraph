//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include <functional>
#include <numeric>

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/pass/implicit_broadcast_elimination.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/provenance.hpp"
#include "ngraph/slice_plan.hpp"
#include "ngraph/type.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

namespace
{
    template <typename OpV0, typename OpV1>
    shared_ptr<Node> op_cast_binary_elementwise_node(const shared_ptr<OpV1>& node)
    {
        const auto input_arg0 = node->input_value(0);
        const auto input_arg1 = node->input_value(1);
        const auto autob = node->get_autob();
        auto replacement_node = make_shared<OpV0>(input_arg0, input_arg1, autob);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    template <typename OpV0, typename OpV1>
    shared_ptr<Node> op_cast_reduction_node(const shared_ptr<OpV1>& node)
    {
        auto replacement_node = make_shared<OpV0>(node->input_value(0), node->input_value(1));
        if (node->get_keep_dims())
        {
            string v1_op_name = string{node->get_type_name()} + ":v1";
            string v0_op_name = string{OpV0{}.get_type_name()} + ":v0";

            NGRAPH_CHECK(node->reduction_axes_constant(),
                         "Unable to convert ",
                         v1_op_name,
                         "to ",
                         v0_op_name,
                         " if reduction axes are not constant (for keep_dims=true). Node: ",
                         *node);
            auto output_pshape = replacement_node->get_output_partial_shape(0);
            NGRAPH_CHECK(output_pshape.is_static(),
                         "Unable to convert ",
                         v1_op_name,
                         "to ",
                         v0_op_name,
                         " if output shape is dynamic (for keep_dims=true). Node: ",
                         *node);
            const auto output_shape = output_pshape.to_shape();
            auto reshaped_output_shape = output_shape;
            for (const auto& axis : node->get_reduction_axes())
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
        return replacement_node;
    }

    // Default is that we did nothing
    shared_ptr<Node> op_cast(shared_ptr<Node> node) { return nullptr; }
    shared_ptr<Node> op_cast(shared_ptr<op::v1::Add> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Add, op::v1::Add>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::AvgPool> node)
    {
        auto const input_arg = node->input_value(0);
        const auto ceil_mode = static_cast<bool>(node->get_rounding_type());
        const auto include_padding_in_avg_computation = !node->get_exclude_pad();
        const auto pad_type = node->get_auto_pad();
        const auto padding_below = node->get_pads_begin();
        const auto padding_above = node->get_pads_end();
        const auto window_movement_strides = node->get_strides();
        const auto window_shape = node->get_kernel();

        auto replacement_node = make_shared<op::v0::AvgPool>(input_arg,
                                                             window_shape,
                                                             window_movement_strides,
                                                             padding_below,
                                                             padding_above,
                                                             include_padding_in_avg_computation,
                                                             pad_type,
                                                             ceil_mode);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::AvgPoolBackprop> node)
    {
        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant());
        const auto forward_arg_shape =
            static_pointer_cast<op::Constant>(node->input_value(1).get_node_shared_ptr())
                ->get_shape_val();
        const auto delta = node->input_value(0);
        const auto include_padding_in_avg_computation = !node->get_exclude_pad();
        const auto padding_below = node->get_pads_begin();
        const auto padding_above = node->get_pads_end();
        const auto window_movement_strides = node->get_strides();
        const auto window_shape = node->get_kernel();

        auto replacement_node =
            make_shared<op::v0::AvgPoolBackprop>(forward_arg_shape,
                                                 delta,
                                                 window_shape,
                                                 window_movement_strides,
                                                 padding_below,
                                                 padding_above,
                                                 include_padding_in_avg_computation);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Broadcast> node)
    {
        auto arg = node->input_value(0);
        auto arg_pshape = arg.get_partial_shape();
        auto arg_rank = arg_pshape.rank();
        auto target_shape_input = node->input_value(1);

        shared_ptr<Node> replacement_node;

        if (arg_rank.is_static() && arg_rank.get_length() == 0 &&
            !target_shape_input.get_node_shared_ptr()->is_constant())
        {
            replacement_node = make_shared<op::DynBroadcast>(
                arg,
                target_shape_input,
                make_shared<op::Range>(make_zero(element::i64, {}),
                                       make_shared<op::ShapeOf>(target_shape_input),
                                       make_constant_from_string("1", element::i64, {})));
        }
        else
        {
            NGRAPH_CHECK(arg_pshape.is_static(),
                         "Unable to convert Broadcast:v1 to Broadcast:v0 "
                         "if argument shape is not static. Node: ",
                         *node);
            const auto& arg_shape = arg_pshape.to_shape();

            NGRAPH_CHECK(target_shape_input.get_node_shared_ptr()->is_constant());
            auto target_shape = node->get_output_shape(0);
            NGRAPH_CHECK(node->get_broadcast_axes().first);

            // (Re)construct axes_mapping.
            AxisSet broadcast_axes = node->get_broadcast_axes().second;
            std::vector<size_t> axes_mapping{
                ngraph::builder::opset1::get_axes_mapping(target_shape, broadcast_axes)};

            Output<Node> squeezed_arg = arg;
            // Collect axes to squeeze. Broadcast v0 "adds" new axes, thus we have to squeeze
            // the empty ones (dim:=1), which would be broadcasted by Broadcast v1.
            std::vector<size_t> empty_axes;
            for (size_t a{0}; a < axes_mapping.size(); ++a)
            {
                if (arg_shape.at(a) == 1 && target_shape.at(axes_mapping.at(a)) != 1)
                {
                    empty_axes.push_back(a);
                }
            }
            // Check if arg_shape contains some more empty dimensions marked to broadcast.
            // If axes_mapping size is less than arg_shape size, then some of arg dimensions may
            // be equal to one and marked to broadcast.
            if (axes_mapping.size() < arg_shape.size())
            {
                for (size_t a{axes_mapping.size()}; a < arg_shape.size(); ++a)
                {
                    if (arg_shape.at(a) == 1)
                    {
                        empty_axes.push_back(a);
                    }
                }
            }
            if (!empty_axes.empty())
            {
                squeezed_arg = builder::squeeze(arg, empty_axes);
            }

            replacement_node =
                make_shared<op::v0::Broadcast>(squeezed_arg, target_shape, broadcast_axes);
        }
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Convolution> node)
    {
        const auto data_arg = node->input_value(0);
        const auto filters_arg = node->input_value(1);
        const auto strides = node->get_strides();
        const size_t num_spatial_dims = strides.size();
        auto replacement_node = make_shared<op::v0::Convolution>(data_arg,
                                                                 filters_arg,
                                                                 node->get_strides(),
                                                                 node->get_dilations(),
                                                                 node->get_pads_begin(),
                                                                 node->get_pads_end(),
                                                                 Strides(num_spatial_dims, 1),
                                                                 node->get_auto_pad());
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::ConvolutionBackpropData> node)
    {
        const auto data_arg = node->input(0).get_source_output();
        const auto filters_arg = node->input(1).get_source_output();

        auto data_pshape = data_arg.get_partial_shape();
        auto filters_pshape = filters_arg.get_partial_shape();

        NGRAPH_CHECK(data_pshape.rank().is_static() && data_pshape[0].is_static() &&
                         filters_pshape.rank().is_static() && filters_pshape[1].is_static(),
                     "Unable to convert ConvolutionBackpropData:v1 to ConvolutionBackpropData:v0 "
                     "if data shape N and filters shape C dimensions are not static. Node: ",
                     *node);

        const size_t num_spatial_dims = data_pshape.rank().get_length() - 2;

        const PartialShape output_pshape{node->output(0).get_partial_shape()};
        NGRAPH_CHECK(output_pshape.is_static(),
                     "Unable to convert ConvolutionBackpropData:v1 to ConvolutionBackpropData:v0 "
                     "if output shape is dynamic. Node: ",
                     *node);
        Shape output_shape = output_pshape.to_shape();

        auto replacement_node =
            make_shared<op::v0::ConvolutionBackpropData>(output_shape,
                                                         filters_arg,
                                                         data_arg,
                                                         node->get_strides(),
                                                         node->get_dilations(),
                                                         node->get_pads_begin(),
                                                         node->get_pads_end(),
                                                         Strides(num_spatial_dims, 1));
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::ConvolutionBackpropFilters> node)
    {
        NGRAPH_CHECK(node->input_value(2).get_node_shared_ptr()->is_constant());
        auto filters_shape =
            static_pointer_cast<op::Constant>(node->input_value(2).get_node_shared_ptr())
                ->get_shape_val();
        const auto data_arg = node->input_value(0);
        const auto delta_arg = node->input_value(1);
        const auto strides = node->get_strides();
        const size_t num_spatial_dims = strides.size();
        auto replacement_node =
            make_shared<op::v0::ConvolutionBackpropFilters>(data_arg,
                                                            filters_shape,
                                                            delta_arg,
                                                            node->get_strides(),
                                                            node->get_dilations(),
                                                            node->get_pads_begin(),
                                                            node->get_pads_end(),
                                                            Strides(num_spatial_dims, 1));
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Divide> node)
    {
        const auto input_arg0 = node->input_value(0);
        const auto input_arg1 = node->input_value(1);
        const auto autob = node->get_autob();
        const bool pydiv = node->is_pythondiv();
        auto replacement_node = make_shared<op::v0::Divide>(input_arg0, input_arg1, pydiv, autob);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Reshape> node)
    {
        shared_ptr<Node> replacement_node;

        const auto target_shape_input = node->input_value(1).get_node_shared_ptr();
        const auto input_rank = node->get_input_partial_shape(0).rank();
        if (target_shape_input->is_constant() && node->get_output_partial_shape(0).is_static() &&
            input_rank.is_static())
        {
            const auto output_shape = node->get_output_shape(0);
            replacement_node = make_shared<op::Reshape>(
                node->input_value(0), get_default_order(input_rank.get_length()), output_shape);
        }
        else
        {
            replacement_node = make_shared<op::v0::DynReshape>(
                node->input_value(0), node->input_value(1), node->get_special_zero());
        }

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Equal> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Equal, op::v1::Equal>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Gather> node)
    {
        auto axis_node = as_type_ptr<op::Constant>(node->input_value(2).get_node_shared_ptr());

        NGRAPH_CHECK(axis_node,
                     "Unable to convert Gather:v1 to Gather:v0 if axis is not constant. Node: ",
                     *node);

        NGRAPH_CHECK(
            axis_node->get_element_type() == element::i64,
            "Unable to convert Gather:v1 to Gather:v0 with axis other type than int64. Node: ",
            *node);

        int64_t axis = axis_node->get_vector<int64_t>()[0];

        auto replacement_node =
            make_shared<op::v0::Gather>(node->input_value(0), node->input_value(1), axis);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::GenerateMask> node)
    {
        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant());
        auto mask_shape =
            static_pointer_cast<op::Constant>(node->input_value(1).get_node_shared_ptr())
                ->get_shape_val();
        auto seed = node->get_seed();
        auto use_seed = node->get_use_seed();
        auto probability = node->get_probability();
        auto et = node->get_element_type();

        auto replacement_node = make_shared<op::v0::GenerateMask>(
            node->input_value(0), mask_shape, et, seed, probability, use_seed);

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Greater> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Greater, op::v1::Greater>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::GreaterEqual> node)
    {
        return op_cast_binary_elementwise_node<op::v0::GreaterEq, op::v1::GreaterEqual>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::GroupConvolution> node)
    {
        const auto data_arg = node->input_value(0);
        const auto filters_arg = node->input_value(1);
        const auto strides = node->get_strides();
        const size_t num_spatial_dims = strides.size();
        auto replacement_node = make_shared<op::GroupConvolution>(data_arg,
                                                                  filters_arg,
                                                                  node->get_strides(),
                                                                  node->get_dilations(),
                                                                  node->get_pads_begin(),
                                                                  node->get_pads_end(),
                                                                  Strides(num_spatial_dims, 1),
                                                                  node->get_auto_pad());
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::GroupConvolutionBackpropData> node)
    {
        const auto data_arg = node->input_value(0);
        const auto filters_arg = node->input_value(1);

        NGRAPH_CHECK(data_arg.get_partial_shape().is_static(),
                     "Unable to convert GroupConvolutionBackpropData:1 to "
                     "GroupConvolutionBackpropData:0 with dynamic data shape. Node: ",
                     *node);

        NGRAPH_CHECK(filters_arg.get_partial_shape().is_static(),
                     "Unable to convert GroupConvolutionBackpropData:1 to "
                     "GroupConvolutionBackpropData:0 with dynamic filters shape. Node: ",
                     *node);

        auto filters_shape = filters_arg.get_shape();
        const size_t groups = filters_shape.at(0);

        const PartialShape output_pshape{node->output(0).get_partial_shape()};
        NGRAPH_CHECK(output_pshape.is_static(),
                     "Unable to convert GroupConvolutionBackpropData:v1 to "
                     "GroupConvolutionBackpropData:v0 "
                     "if output_shape is dynamic. Node: ",
                     *node);
        Shape output_shape = output_pshape.to_shape();

        // Convert filters data layout from [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
        // into [C x M/group x k1 x k2 x ... x kn]
        filters_shape.erase(filters_shape.begin());
        filters_shape[0] *= groups;

        auto reshaped_filters = builder::opset1::reshape(node->input_value(1), filters_shape);

        auto replacement_node = make_shared<op::v0::GroupConvolutionBackpropData>(
            op::Constant::create(data_arg.get_element_type(), output_shape, {0}),
            reshaped_filters,
            data_arg,
            node->get_strides(),
            node->get_dilations(),
            node->get_pads_begin(),
            node->get_pads_end(),
            groups);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Less> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Less, op::v1::Less>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::LessEqual> node)
    {
        return op_cast_binary_elementwise_node<op::v0::LessEq, op::v1::LessEqual>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::LogicalAnd> node)
    {
        return op_cast_binary_elementwise_node<op::v0::And, op::v1::LogicalAnd>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::LogicalNot> node)
    {
        auto replacement_node = make_shared<op::v0::Not>(node->input_value(0));
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::LogicalOr> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Or, op::v1::LogicalOr>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::LogicalXor> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Xor, op::v1::LogicalXor>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Maximum> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Maximum, op::v1::Maximum>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::MaxPool> node)
    {
        auto const input_arg = node->input_value(0);
        auto ceil_mode = static_cast<bool>(node->get_rounding_type());
        auto pad_type = node->get_auto_pad();
        auto padding_below = node->get_pads_begin();
        auto padding_above = node->get_pads_end();
        auto window_movement_strides = node->get_strides();
        auto window_shape = node->get_kernel();

        auto replacement_node = make_shared<op::v0::MaxPool>(input_arg,
                                                             window_shape,
                                                             window_movement_strides,
                                                             padding_below,
                                                             padding_above,
                                                             pad_type,
                                                             ceil_mode);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::MaxPoolBackprop> node)
    {
        const auto padding_below = node->get_pads_begin();
        const auto padding_above = node->get_pads_end();
        const auto window_movement_strides = node->get_strides();
        const auto window_shape = node->get_kernel();

        const auto arg_forward = node->input_value(0);
        const auto delta = node->input_value(1);

        shared_ptr<Node> replacement_node;
        if (node->get_input_size() == 3)
        {
            const auto result_forward = node->input_value(2);
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
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Minimum> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Minimum, op::v1::Minimum>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Multiply> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Multiply, op::v1::Multiply>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::NotEqual> node)
    {
        return op_cast_binary_elementwise_node<op::v0::NotEqual, op::v1::NotEqual>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::OneHot> node)
    {
        const auto indices = node->input_value(0);
        const auto depth = node->input_value(1).get_node_shared_ptr();
        auto on_value = node->input_value(2);
        auto off_value = node->input_value(3);
        const auto axis = node->get_axis();

        NGRAPH_CHECK(depth->is_constant(), "depth input must be constant", *node);
        const auto output_pshape = node->get_output_partial_shape(0);
        NGRAPH_CHECK(output_pshape.is_static(), "output shape must be static", *node);
        const auto output_shape = output_pshape.to_shape();

        auto one_hot = std::make_shared<ngraph::op::Convert>(
            std::make_shared<ngraph::op::OneHot>(indices, output_shape, axis),
            on_value.get_element_type());

        auto broadcasted_values = builder::numpy_broadcast_outputs({one_hot, on_value, off_value});
        on_value = broadcasted_values[1];
        off_value = broadcasted_values[2];

        auto replacement_node = one_hot * (on_value - off_value) + off_value;

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Pad> node)
    {
        const auto pad_arg = node->input_value(0);
        Output<Node> pad_value;
        if (node->get_input_size() == 4)
        {
            pad_value = node->input_value(3);
        }
        else
        {
            pad_value =
                make_shared<op::Constant>(pad_arg.get_element_type(), Shape{}, vector<float>{0.f});
        }
        auto replacement_node = make_shared<op::v0::Pad>(
            pad_arg, pad_value, node->get_pads_begin(), node->get_pads_end(), node->get_pad_mode());

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Power> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Power, op::v1::Power>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::ReduceMax> node)
    {
        return op_cast_reduction_node<op::v0::Max, op::v1::ReduceMax>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::ReduceMin> node)
    {
        return op_cast_reduction_node<op::v0::Min, op::v1::ReduceMin>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::ReduceProd> node)
    {
        return op_cast_reduction_node<op::v0::Product, op::v1::ReduceProd>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::ReduceSum> node)
    {
        return op_cast_reduction_node<op::v0::Sum, op::v1::ReduceSum>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Reverse> node)
    {
        auto axes_node = node->input_value(1).get_node_shared_ptr();
        NGRAPH_CHECK(axes_node->is_constant(),
                     "Unable to convert Reverse:v1 to Reverse:v0 "
                     "if reduction axes are not constant. Node: ",
                     *node);
        const auto axes_node_const = as_type_ptr<op::Constant>(axes_node);
        AxisSet axes{};
        if (node->get_mode() == op::v1::Reverse::Mode::INDEX)
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
        auto replacement_node = make_shared<op::v0::Reverse>(node->input_value(0), axes);

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Select> node)
    {
        ngraph::pass::ImplicitBroadcastElimination().run_on_node(node);
        auto replacement_node = make_shared<op::v0::Select>(
            node->input_value(0), node->input_value(1), node->input_value(2));
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::StridedSlice> node)
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

        SlicePlan p = make_slice_plan(input_data_pshape.to_shape(),
                                      begin_const->get_vector<int64_t>(),
                                      end_const->get_vector<int64_t>(),
                                      strides->get_vector<int64_t>(),
                                      convert_mask_to_axes(node->get_begin_mask()),
                                      convert_mask_to_axes(node->get_end_mask()),
                                      convert_mask_to_axes(node->get_new_axis_mask()),
                                      convert_mask_to_axes(node->get_shrink_axis_mask()),
                                      convert_mask_to_axes(node->get_ellipsis_mask()));

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
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Softmax> node)
    {
        auto axis = node->get_axis();
        auto data = node->input(0);
        auto data_shape = data.get_shape();
        std::vector<size_t> axes(data_shape.size() - axis);
        std::iota(std::begin(axes), std::end(axes), axis);
        auto replacement_node = make_shared<op::v0::Softmax>(node->input_value(0), axes);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Split> node)
    {
        const auto num_splits = node->get_num_splits();

        auto replacement_node =
            make_shared<op::v0::Split>(node->input_value(0), node->input_value(1), num_splits);

        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Subtract> node)
    {
        return op_cast_binary_elementwise_node<op::v0::Subtract, op::v1::Subtract>(node);
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::TopK> node)
    {
        const auto axis = node->get_axis();
        const auto sort_type = node->get_sort_type();
        const auto index_elem_type = node->get_index_element_type();

        bool compute_max;
        switch (node->get_mode())
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
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::Transpose> node)
    {
        const auto data = node->input_value(0);

        const auto data_pshape = data.get_partial_shape();
        NGRAPH_CHECK(data_pshape.is_static(),
                     "Unable to convert Transpose:v1 to Reshape:v0 "
                     "if data shape is dynamic. Node: ",
                     *node);
        const auto data_shape = data_pshape.to_shape();

        const auto order_node = node->input_value(1).get_node_shared_ptr();
        NGRAPH_CHECK(order_node->is_constant(),
                     "Unable to convert Transpose:v1 to Reshape:v0 "
                     "if order node is not constant. Node: ",
                     *node);
        const auto order_const = as_type_ptr<op::Constant>(order_node);

        auto order = order_const->get_axis_vector_val();
        Shape out_shape = data_shape;
        if (order.empty())
        {
            order.resize(out_shape.size());
            iota(begin(order), end(order), 0);
        }
        else
        {
            for (size_t i = 0; i < order.size(); ++i)
            {
                out_shape[i] = data_shape.at(order.at(i));
            }
        }

        auto replacement_node = make_shared<op::v0::Reshape>(data, order, out_shape);
        replace_node(node, replacement_node);
        return replacement_node;
    }

    shared_ptr<Node> op_cast(shared_ptr<op::v1::VariadicSplit> node)
    {
        const auto split_lengths = node->input_value(2).get_node_shared_ptr();

        NGRAPH_CHECK(split_lengths->is_constant(),
                     "Unable to convert VariadicSplit:v1 to Split:v0 "
                     "if 'split_lengths' input is not constant. Node: ",
                     *node);

        const auto splits = as_type_ptr<op::Constant>(split_lengths)->get_vector<int64_t>();
        const std::vector<size_t> splits_unsigned{splits.begin(), splits.end()};

        auto replacement_node =
            make_shared<op::v0::Split>(node->input_value(0), node->input_value(1), splits_unsigned);

        replace_node(node, replacement_node);
        return replacement_node;
    }

    using DispatchMap = map<NodeTypeInfo, std::function<bool(shared_ptr<Node> node)>>;

    template <typename T>
    bool op_cast_thunk(shared_ptr<Node> node)
    {
        auto downgraded_node = op_cast(as_type_ptr<T>(node));
        if (downgraded_node)
        {
            if (ngraph::get_provenance_enabled())
            {
                const std::string provenance_tag =
                    "<Opset0_Downgrade (v1 " + std::string(node->get_type_name()) + ")>";
                downgraded_node->add_provenance_tags_above(node->input_values(), {provenance_tag});
            }
            return true;
        }
        return false;
    }

    DispatchMap& get_dispatch_map()
    {
        static DispatchMap dispatch_map{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, op_cast_thunk<NAMESPACE::NAME>},
#include "ngraph/opsets/opset1_tbl.hpp"
            NGRAPH_OP(AvgPoolBackprop, op::v1) NGRAPH_OP(ConvolutionBackpropFilters, op::v1)
                NGRAPH_OP(GenerateMask, op::v1) NGRAPH_OP(MaxPoolBackprop, op::v1)
#undef NGRAPH_OP
        };
        return dispatch_map;
    }
} // namespace

bool pass::Opset0Downgrade::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;
    auto& dispatch_map = get_dispatch_map();
    auto it = dispatch_map.find(node->get_type_info());
    if (it != dispatch_map.end())
    {
        modified = it->second(node);
    }
    return modified;
}
