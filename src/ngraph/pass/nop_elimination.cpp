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

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/fused/squeeze.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"
#include "ngraph/op/non_zero.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/util.hpp"
#include "nop_elimination.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) x::type_info

static bool eliminate_nop(const std::shared_ptr<Node>& node)
{
    // skip if shapes are dynamic
    if (node->get_input_partial_shape(0).is_dynamic() ||
        node->get_output_partial_shape(0).is_dynamic())
    {
        return false;
    }

    if (node->get_input_shape(0) == node->get_output_shape(0))
    {
        return replace_output_update_name(node->output(0), node->input_value(0));
    }
    return false;
}

static bool eliminate_sum(const std::shared_ptr<Node>& node)
{
    auto sum = as_type_ptr<op::v0::Sum>(node);
    if (sum->get_reduction_axes().empty())
    {
        return replace_output_update_name(node->output(0), node->input_value(0));
    }
    return false;
}

static bool eliminate_convert(const std::shared_ptr<Node>& node)
{
    bool is_out_type_agnostic = false;
    static const std::set<NodeTypeInfo> type_agnostic{TI(opset3::NonZero)};
    if (node->output(0).get_target_inputs().size() == 1)
    {
        Input<Node> out = *node->output(0).get_target_inputs().begin();
        is_out_type_agnostic = type_agnostic.count(out.get_node()->get_type_info()) == 1;
    }
    auto convert = as_type_ptr<opset3::Convert>(node);
    auto input = convert->input_value(0);
    if (convert->get_convert_element_type() == input.get_element_type() || is_out_type_agnostic)
    {
        if (is_out_type_agnostic && is_type<opset3::Convert>(input.get_node()))
        {
            input = input.get_node()->input_value(0);
        }
        return replace_output_update_name(node->output(0), input);
    }
    return false;
}

static bool eliminate_concat(const std::shared_ptr<Node>& node)
{
    auto node_input = node->input_value(0);

    // remove concat with single input
    if (node->get_input_size() == 1)
    {
        return replace_output_update_name(node->output(0), node_input);
    }
    return false;
}

static bool eliminate_reshape_v1(const std::shared_ptr<Node>& node)
{
    auto input = node->input_value(0);
    // check if reshape is not identity op
    if (input.get_partial_shape().is_dynamic() || node->get_output_partial_shape(0).is_dynamic())
    {
        NGRAPH_DEBUG << node << " has dynamic shapes.";
        return false;
    }
    // remove identity op
    if (input.get_shape() == node->get_output_shape(0))
    {
        return replace_output_update_name(node->output(0), input);
    }
    // eliminate redundant reshape, squeeze, or unsqueeze
    if (is_type<opset3::Squeeze>(input.get_node()) ||
        is_type<opset3::Unsqueeze>(input.get_node()) || is_type<opset3::Reshape>(input.get_node()))
    {
        auto shape = node->get_output_shape(0);
        std::vector<int64_t> vi;
        vi.assign(shape.begin(), shape.end());
        auto pat = op::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
        auto new_reshape =
            make_shared<opset3::Reshape>(input.get_node()->input_value(0), pat, false);
        return replace_output_update_name(node->output(0), new_reshape->output(0));
    }

    return false;
}

static bool eliminate_unsqueeze(const std::shared_ptr<Node>& node)
{
    auto unsqueeze = as_type_ptr<opset3::Unsqueeze>(node);
    auto input = unsqueeze->input_value(0).get_node_shared_ptr();
    auto squeeze = as_type_ptr<opset3::Squeeze>(input);
    // eliminate redundant squeeze->unsqueeze
    if (auto squeeze = as_type_ptr<opset3::Squeeze>(input))
    {
        if (!ngraph::compare_constants(squeeze->input_value(1).get_node_shared_ptr(),
                                       unsqueeze->input_value(1).get_node_shared_ptr()))
        {
            NGRAPH_DEBUG << "squeeze->unsqueeze axes do not match";
            return false;
        }
        return replace_output_update_name(unsqueeze->output(0), squeeze->input_value(0));
    }

    // eliminate redundant unsqueeze
    if (as_type_ptr<opset3::Reshape>(input) && !node->get_output_partial_shape(0).is_dynamic())
    {
        auto shape = node->get_shape();
        std::vector<int64_t> vi;
        vi.assign(shape.begin(), shape.end());
        auto pat = op::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
        auto new_reshape =
            make_shared<opset3::Reshape>(input->input_value(0).get_node_shared_ptr(), pat, false);
        return replace_output_update_name(node->output(0), new_reshape->output(0));
    }
    return false;
}

static bool eliminate_squeeze(const std::shared_ptr<Node>& node)
{
    auto squeeze = as_type_ptr<opset3::Squeeze>(node);
    auto input = squeeze->input_value(0).get_node_shared_ptr();
    // eliminate redundant unsqueeze->squeeze
    if (auto unsqueeze = as_type_ptr<opset3::Unsqueeze>(input))
    {
        if (!ngraph::compare_constants(unsqueeze->input_value(1).get_node_shared_ptr(),
                                       squeeze->input_value(1).get_node_shared_ptr()))
        {
            NGRAPH_DEBUG << "unsqueeze->squeeze axes do not match";
            return false;
        }
        return replace_output_update_name(squeeze->output(0), unsqueeze->input_value(0));
    }
    // eliminate redundant squeeze
    if (as_type_ptr<opset3::Reshape>(input) && !node->get_output_partial_shape(0).is_dynamic())
    {
        auto shape = node->get_shape();
        std::vector<int64_t> vi;
        vi.assign(shape.begin(), shape.end());
        auto pat = op::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
        auto new_reshape =
            make_shared<opset3::Reshape>(input->input_value(0).get_node_shared_ptr(), pat, false);
        return replace_output_update_name(node->output(0), new_reshape->output(0));
    }
    return false;
}

static bool eliminate_stop_gradient(const std::shared_ptr<Node>& node)
{
    replace_output_update_name(node->output(0), node->input_value(0));
    return true;
}

static const std::unordered_map<NodeTypeInfo, std::function<bool(const std::shared_ptr<Node>&)>>
    dispatcher{{TI(op::v0::Pad), &eliminate_nop},
               {TI(opset3::Pad), &eliminate_nop},
               {TI(op::v0::Sum), &eliminate_sum},
               {TI(opset3::Convert), &eliminate_convert},
               {TI(op::v0::Slice), &eliminate_nop},
               {TI(op::v0::StopGradient), &eliminate_stop_gradient},
               {TI(opset3::Reshape), &eliminate_reshape_v1},
               {TI(opset3::Concat), &eliminate_concat},
               {TI(opset3::Squeeze), &eliminate_squeeze},
               {TI(opset3::Unsqueeze), &eliminate_unsqueeze},
               {TI(op::v0::Broadcast), &eliminate_nop}};

bool pass::NopElimination::run_on_function(std::shared_ptr<Function> function)
{
    bool clobbered = false;

    for (const auto& n : function->get_ops())
    {
        // Work around a warning [-Wpotentially-evaluated-expression]
        const Node& node = *n;
        auto handler = dispatcher.find(node.get_type_info());
        if (handler != dispatcher.end())
        {
            clobbered = handler->second(n) || clobbered;
        }
    }

    return clobbered;
}
