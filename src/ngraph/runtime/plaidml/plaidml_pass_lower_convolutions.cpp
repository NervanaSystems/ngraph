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

#include <numeric>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/plaidml/plaidml_ops_convolution.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_lower_convolutions.hpp"

ngraph::runtime::plaidml::pass::LowerConvolutions::LowerConvolutions()
{
    auto convolution_op =
        std::make_shared<pattern::op::Label>(element::i8, Shape{}, [](std::shared_ptr<Node> node) {
            return pattern::has_class<ngraph::op::Convolution>()(node) ||
                   pattern::has_class<ngraph::op::ConvolutionBackpropData>()(node) ||
                   pattern::has_class<ngraph::op::ConvolutionBackpropFilters>()(node);
        });

    auto callback = [](pattern::Matcher& m) {
        auto to_transpose = [](const std::shared_ptr<Node>& node) -> ngraph::op::Reshape* {
            if (!node)
            {
                return nullptr;
            }
            auto* reshape = dynamic_cast<ngraph::op::Reshape*>(node.get());
            if (reshape && reshape->get_is_transpose())
            {
                return reshape;
            }
            return nullptr;
        };

        auto to_axes = [](const std::shared_ptr<Node>& node, ngraph::op::Reshape* reshape) {
            if (reshape)
            {
                return reshape->get_input_order();
            }
            return get_default_order(node->get_shape());
        };

        std::shared_ptr<Node> node = m.get_match_root();

        std::shared_ptr<Node> output;
        auto users = node->get_users(true);
        if (users.size() == 1)
        {
            output = users[0];
        }
        auto target = node;
        auto* output_transpose = to_transpose(output);
        if (output_transpose)
        {
            target = output;
        }
        // N.B. For the output axes, we can either use the convolution
        // or the final output op -- but there might not be an output
        // op.  Using target always works.
        AxisVector out_axes = to_axes(target, output_transpose);

        auto lhs = node->get_argument(0);
        auto* lhs_transpose = to_transpose(lhs);
        if (lhs_transpose)
        {
            lhs = lhs_transpose->get_argument(0);
        }
        AxisVector lhs_axes = to_axes(lhs, lhs_transpose);

        auto rhs = node->get_argument(1);
        auto* rhs_transpose = to_transpose(rhs);
        if (rhs_transpose)
        {
            rhs = rhs_transpose->get_argument(0);
        }
        AxisVector rhs_axes = to_axes(rhs, rhs_transpose);

        {
            auto conv = as_type_ptr<ngraph::op::Convolution>(node);
            if (conv)
            {
                replace_node(target,
                             std::make_shared<plaidml::op::Convolution>(conv,
                                                                        OutputVector{lhs, rhs},
                                                                        std::move(lhs_axes),
                                                                        std::move(rhs_axes),
                                                                        std::move(out_axes)));
                return true;
            }
        }

        {
            auto conv_bp_data = as_type_ptr<ngraph::op::ConvolutionBackpropData>(node);
            if (conv_bp_data)
            {
                replace_node(
                    target,
                    std::make_shared<plaidml::op::ConvolutionBackpropData>(conv_bp_data,
                                                                           OutputVector{lhs, rhs},
                                                                           std::move(lhs_axes),
                                                                           std::move(rhs_axes),
                                                                           std::move(out_axes)));
                return true;
            }
        }

        {
            auto conv_bp_filters = as_type_ptr<ngraph::op::ConvolutionBackpropFilters>(node);
            if (conv_bp_filters)
            {
                replace_node(target,
                             std::make_shared<plaidml::op::ConvolutionBackpropFilters>(
                                 conv_bp_filters,
                                 OutputVector{lhs, rhs},
                                 std::move(lhs_axes),
                                 std::move(rhs_axes),
                                 std::move(out_axes)));
                return true;
            }
        }

        return false;
    };

    add_matcher(std::make_shared<pattern::Matcher>(convolution_op), callback);
}
