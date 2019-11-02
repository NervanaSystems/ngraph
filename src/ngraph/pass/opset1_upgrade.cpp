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
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/strided_slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/topk.hpp"

#include <limits>
#include <numeric>

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

bool pass::Opset1Upgrade::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;

    size_t op_version = node->get_version();

    if (op_version == 1)
    {
        return modified;
    }

    NGRAPH_CHECK(op_version == 0,
                 "Op version 1 transformation pass failed for ",
                 *node,
                 ", only op version 0 operations expected. Op version ",
                 op_version,
                 " found.");

// Not all enumeration values explicitly handled in switch
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (get_typeid(node))
    {
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
            make_shared<op::v1::ConvolutionBackpropData>(data_batch_shape,
                                                         node->input(0).get_source_output(),
                                                         node->input(1).get_source_output(),
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
                                                            filters_shape,
                                                            node->input(1).get_source_output(),
                                                            strides,
                                                            dilations,
                                                            pads_begin,
                                                            pads_end);
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
