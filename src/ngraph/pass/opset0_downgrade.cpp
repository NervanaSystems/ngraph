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
#include <functional>
#include <numeric>

#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/slice_plan.hpp"
#include "ngraph/type.hpp"

using namespace std;
using namespace ngraph;

namespace
{
    template <typename OpV0, typename OpV1>
    void op_cast_binary_elementwise_node(const shared_ptr<OpV1>& node)
    {
        const auto input_arg0 = node->input_value(0);
        const auto input_arg1 = node->input_value(1);
        const auto autob = node->get_autob();
        auto replacement_node = make_shared<OpV0>(input_arg0, input_arg1, autob);
        replace_node(node, replacement_node);
    }

    // Default is that we didn nothing
    bool op_cast(shared_ptr<Node> node) { return false; }
    bool op_cast(shared_ptr<op::v1::Add> node)
    {
        op_cast_binary_elementwise_node<op::v0::Add, op::v1::Add>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::AvgPool> node)
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
        return true;
    }

    bool op_cast(shared_ptr<op::v1::AvgPoolBackprop> node)
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
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Broadcast> node)
    {
        auto arg = node->input_value(0);
        NGRAPH_CHECK(node->input_value(1).get_node_shared_ptr()->is_constant());
        auto target_shape =
            static_pointer_cast<op::Constant>(node->input_value(1).get_node_shared_ptr())
                ->get_shape_val();
        NGRAPH_CHECK(node->get_broadcast_axes().first);
        auto replacement_node =
            make_shared<op::v0::Broadcast>(arg, target_shape, node->get_broadcast_axes().second);

        replace_node(node, replacement_node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Convolution> node)
    {
        const auto data_arg = node->input_value(0);
        const auto filters_arg = node->input_value(1);
        const PartialShape& data_arg_pshape = node->get_input_partial_shape(0);
        NGRAPH_CHECK(data_arg_pshape.rank().is_static(),
                     "Unable to convert Convolution:v1 to Convolution:v0 if data argument "
                     "rank is dynamic. Node: ",
                     *node);
        const size_t num_spatial_dims = static_cast<size_t>(data_arg_pshape.rank()) - 2;
        auto replacement_node = make_shared<op::v0::Convolution>(data_arg,
                                                                 filters_arg,
                                                                 node->get_strides(),
                                                                 node->get_dilations(),
                                                                 node->get_pads_begin(),
                                                                 node->get_pads_end(),
                                                                 Strides(num_spatial_dims, 1),
                                                                 node->get_auto_pad());
        replace_node(node, replacement_node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::ConvolutionBackpropData> node)
    {
        auto output_shape = as_type_ptr<op::Constant>(node->input_value(2).get_node_shared_ptr());
        const auto data_arg = node->input(0).get_source_output();
        const auto filters_arg = node->input(1).get_source_output();
        const PartialShape& delta_arg_pshape = node->get_input_partial_shape(1);
        NGRAPH_CHECK(delta_arg_pshape.rank().is_static(),
                     "Unable to convert ConvolutionBackpropData:v1 to ConvolutionBackpropData:v0 "
                     "if delta argument rank is dynamic. Node: ",
                     *node);
        NGRAPH_CHECK(output_shape,
                     "Unable to convert ConvolutionBackpropData:v1 to ConvolutionBackpropData:v0 "
                     "if output_shape is not constant. Node: ",
                     *node);
        const size_t num_spatial_dims = static_cast<size_t>(delta_arg_pshape.rank()) - 2;

        auto output_padding = node->get_output_padding();

        bool is_op_valid = all_of(
            output_padding.begin(), output_padding.end(), [](size_t value) { return value == 0; });

        NGRAPH_CHECK(is_op_valid,
                     "Unable to convert ConvolutionBackpropData:v1 to ConvolutionBackpropData:v0 "
                     "with output padding other than `0`. Node: ",
                     *node);

        auto replacement_node =
            make_shared<op::v0::ConvolutionBackpropData>(output_shape->get_shape_val(),
                                                         filters_arg,
                                                         data_arg,
                                                         node->get_strides(),
                                                         node->get_dilations(),
                                                         node->get_pads_begin(),
                                                         node->get_pads_end(),
                                                         Strides(num_spatial_dims, 1));
        replace_node(node, replacement_node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::ConvolutionBackpropFilters> node)
    {
        NGRAPH_CHECK(node->input_value(2).get_node_shared_ptr()->is_constant());
        auto filters_shape =
            static_pointer_cast<op::Constant>(node->input_value(2).get_node_shared_ptr())
                ->get_shape_val();
        const auto data_arg = node->input_value(0);
        const auto delta_arg = node->input_value(1);
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
                                                            node->get_strides(),
                                                            node->get_dilations(),
                                                            node->get_pads_begin(),
                                                            node->get_pads_end(),
                                                            Strides(num_spatial_dims, 1));
        replace_node(node, replacement_node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Divide> node)
    {
        const auto input_arg0 = node->input_value(0);
        const auto input_arg1 = node->input_value(1);
        const auto autob = node->get_autob();
        const bool pydiv = node->is_pythondiv();
        auto replacement_node = make_shared<op::v0::Divide>(input_arg0, input_arg1, pydiv, autob);
        replace_node(node, replacement_node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Reshape> node)
    {
        auto replacement_node = make_shared<op::v0::DynReshape>(
            node->input_value(0), node->input_value(1), node->get_zero_flag());
        replace_node(node, replacement_node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Equal> node)
    {
        op_cast_binary_elementwise_node<op::v0::Equal, op::v1::Equal>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::GenerateMask> node)
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
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Greater> node)
    {
        op_cast_binary_elementwise_node<op::v0::Greater, op::v1::Greater>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::GreaterEqual> node)
    {
        op_cast_binary_elementwise_node<op::v0::GreaterEq, op::v1::GreaterEqual>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Less> node)
    {
        op_cast_binary_elementwise_node<op::v0::Less, op::v1::Less>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::LessEqual> node)
    {
        op_cast_binary_elementwise_node<op::v0::LessEq, op::v1::LessEqual>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::LogicalAnd> node)
    {
        op_cast_binary_elementwise_node<op::v0::And, op::v1::LogicalAnd>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::LogicalNot> node)
    {
        replace_node(node, make_shared<op::v0::Not>(node->input_value(0)));
        return true;
    }

    bool op_cast(shared_ptr<op::v1::LogicalOr> node)
    {
        op_cast_binary_elementwise_node<op::v0::Or, op::v1::LogicalOr>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::LogicalXor> node)
    {
        op_cast_binary_elementwise_node<op::v0::Xor, op::v1::LogicalXor>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Maximum> node)
    {
        op_cast_binary_elementwise_node<op::v0::Maximum, op::v1::Maximum>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::MaxPool> node)
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
        return true;
    }

    bool op_cast(shared_ptr<op::v1::MaxPoolBackprop> node)
    {
        const auto padding_below = node->get_pads_begin();
        const auto padding_above = node->get_pads_end();
        const auto window_movement_strides = node->get_strides();
        const auto window_shape = node->get_kernel();

        const auto arg_forward = node->input_value(0);
        const auto delta = node->input_value(1);

        shared_ptr<Node> replacement_node;
        if (node->get_inputs().size() == 3)
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
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Minimum> node)
    {
        op_cast_binary_elementwise_node<op::v0::Minimum, op::v1::Minimum>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Multiply> node)
    {
        op_cast_binary_elementwise_node<op::v0::Multiply, op::v1::Multiply>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::NotEqual> node)
    {
        op_cast_binary_elementwise_node<op::v0::NotEqual, op::v1::NotEqual>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::OneHot> node)
    {
        const auto indices = node->input_value(0).get_node_shared_ptr();
        const auto depth = node->input_value(1).get_node_shared_ptr();
        auto on_value = node->input_value(2).get_node_shared_ptr();
        auto off_value = node->input_value(3).get_node_shared_ptr();
        const auto axis = node->get_axis();

        NGRAPH_CHECK(depth->is_constant(), "depth input must be constant", *node);
        const auto const_depth = as_type_ptr<op::Constant>(depth);
        std::int64_t depth_value = const_depth->get_vector<std::int64_t>()[0];

        const auto indices_shape = node->get_input_partial_shape(0);
        NGRAPH_CHECK(indices_shape.is_static(), "indices shape must be static", *node);
        auto output_shape = indices_shape.to_shape();
        output_shape.insert(output_shape.begin() + axis, depth_value);

        auto one_hot = std::make_shared<ngraph::op::Convert>(
            std::make_shared<ngraph::op::OneHot>(indices, output_shape, axis),
            on_value->get_element_type());

        auto broadcasted_values = op::numpy_style_broadcast({one_hot, on_value, off_value});
        on_value = broadcasted_values[1];
        off_value = broadcasted_values[2];

        auto replacement_node = one_hot * (on_value - off_value) + off_value;

        replace_node(node, replacement_node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Pad> node)
    {
        const auto pad_arg = node->input_value(0);
        const auto pad_value = node->input_value(3);
        auto replacement_node = make_shared<op::v0::Pad>(
            pad_arg, pad_value, node->get_pads_begin(), node->get_pads_end(), node->get_pad_mode());

        replace_node(node, replacement_node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Power> node)
    {
        op_cast_binary_elementwise_node<op::v0::Power, op::v1::Power>(node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::ReduceProd> node)
    {
        auto replacement_node =
            make_shared<op::v0::Product>(node->input_value(0), node->input_value(1));
        if (node->get_keep_dims())
        {
            NGRAPH_CHECK(node->reduction_axes_constant(),
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
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Reverse> node)
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
        return true;
    }

    bool op_cast(shared_ptr<op::v1::StridedSlice> node)
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
        return true;
    }

    bool op_cast(shared_ptr<op::v1::Softmax> node)
    {
        auto axis = node->get_axis();
        auto data = node->input(0);
        auto data_shape = data.get_shape();
        std::vector<size_t> axes(data_shape.size() - axis);
        std::iota(std::begin(axes), std::end(axes), axis);
        auto replacement_node = make_shared<op::v0::Softmax>(node->input_value(0), axes);
        replace_node(node, replacement_node);
        return true;
    }

    bool op_cast(shared_ptr<op::v1::ReduceSum> node)
    {
        auto replacement_node =
            make_shared<op::v0::Sum>(node->input_value(0), node->input_value(1));
        if (node->get_keep_dims())
        {
            NGRAPH_CHECK(node->reduction_axes_constant(),
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
        return true;
    }

    bool op_cast(shared_ptr<op::v1::TopK> node)
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
        return true;
    }

    using DispatchMap = map<NodeTypeInfo, std::function<bool(shared_ptr<Node> node)>>;

    template <typename T>
    bool op_cast_thunk(shared_ptr<Node> node)
    {
        return op_cast(as_type_ptr<T>(node));
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
}

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
