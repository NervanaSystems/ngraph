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

#include "ngraph/runtime/cpu/pass/cpu_horizontal_fusion.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"

using namespace ngraph;
using namespace std;

bool has_same_attributes(const std::shared_ptr<ngraph::op::ConvolutionBias> conv1,
                         const std::shared_ptr<ngraph::op::ConvolutionBias> conv2)
{
    auto conv1_shape = conv1->get_input_shape(1);
    auto conv2_shape = conv2->get_input_shape(1);
    if (conv1_shape[2] != conv2_shape[2] || conv1_shape[3] != conv2_shape[3])
    {
        NGRAPH_DEBUG << "conv_horizontal_fusion: skip conv node with different filter shape\n";
        return false;
    }
    if (conv1->get_window_movement_strides() != conv2->get_window_movement_strides())
    {
        NGRAPH_DEBUG << "conv_horizontal_fusion: skip conv node with different window "
                        "movement strides\n";
        return false;
    }
    if (conv1->get_window_dilation_strides() != conv2->get_window_dilation_strides())
    {
        NGRAPH_DEBUG << "conv_horizontal_fusion: skip conv node with different window "
                        "dilation strides\n";
        return false;
    }
    if (conv1->get_padding_below() != conv2->get_padding_below())
    {
        NGRAPH_DEBUG << "conv_horizontal_fusion: skip conv node with different padding "
                        "below\n";
        return false;
    }
    if (conv1->get_padding_above() != conv2->get_padding_above())
    {
        NGRAPH_DEBUG << "conv_horizontal_fusion: skip conv node with different padding "
                        "above\n";
        return false;
    }
    if (conv1->get_data_dilation_strides() != conv2->get_data_dilation_strides())
    {
        NGRAPH_DEBUG << "conv_horizontal_fusion: skip conv node with different data "
                        "dilation strides\n";
        return false;
    }
    if (conv1->with_relu() != conv2->with_relu())
    {
        NGRAPH_DEBUG << "conv_horizontal_fusion: skip conv node with different relu "
                        "status\n";
        return false;
    }
    return true;
};

void ngraph::runtime::cpu::pass::CPUHorizontalFusion::cpu_conv_horizontal_fusion()
{
    auto has_multiple_users = [](std::shared_ptr<Node> n) {
        auto inputs = n->get_output_inputs(0);
        return inputs.size() > 1;
    };

    auto data_conv = std::make_shared<pattern::op::Label>(
        element::f32, Shape{1, 256, 35, 35}, has_multiple_users);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, Shape{64, 256, 1, 1});
    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{64});

    auto conv_bias = std::make_shared<ngraph::op::ConvolutionBias>(data_conv,
                                                                   filters,
                                                                   bias,
                                                                   Strides{1, 1},
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{0, 0},
                                                                   CoordinateDiff{0, 0},
                                                                   Strides{1, 1},
                                                                   true);

    auto callback = [data_conv](pattern::Matcher& m) {
        NGRAPH_DEBUG << "conv_horizontal_fusion: In a callback for conv horizontal fusion for "
                     << m.get_match_root()->get_name();

        auto conv_bias_root = std::static_pointer_cast<op::ConvolutionBias>(m.get_match_root());

        // check if the node has been replaced
        if (conv_bias_root->get_users().empty())
        {
            NGRAPH_DEBUG << "conv_horizontal_fusion: root node has been replaced\n";
            return false;
        }

        // get weights and bias from each CBR and create Concat nodes
        std::vector<std::shared_ptr<Node>> weights_nodes;
        std::vector<std::shared_ptr<Node>> bias_nodes;
        std::vector<std::shared_ptr<Node>> conv_bias_nodes;

        for (auto u : m.get_pattern_map()[data_conv]->get_users())
        {
            if (!is_used(u.get()))
            {
                NGRAPH_DEBUG << "conv_horizontal_fusion: dead node\n";
                continue;
            }
            if (!pattern::has_class<ngraph::op::ConvolutionBias>()(u))
            {
                NGRAPH_DEBUG << "conv_horizontal_fusion: not conv_bias node\n";
                continue;
            }
            if (u->get_argument(0) != m.get_pattern_map()[data_conv])
            {
                NGRAPH_DEBUG << "conv_horizontal_fusion: data_conv is not input 0 for "
                             << u->get_name() << "\n";
                continue;
            }

            auto conv_u = std::static_pointer_cast<op::ConvolutionBias>(u);
            if (!has_same_attributes(conv_u, conv_bias_root))
            {
                continue;
            }

            weights_nodes.push_back(u->get_argument(1));
            bias_nodes.push_back(u->get_argument(2));
            conv_bias_nodes.push_back(u);
        }

        if (conv_bias_nodes.size() <= 1)
        {
            NGRAPH_DEBUG << "conv_horizontal_fusion: need more than one nodes to do fusion\n";
            return false;
        }

        auto concat_weights = std::make_shared<ngraph::op::Concat>(weights_nodes, 0);
        auto concat_bias = std::make_shared<ngraph::op::Concat>(bias_nodes, 0);
        auto conv_bias_new = std::make_shared<ngraph::op::ConvolutionBias>(
            conv_bias_root->get_argument(0),
            concat_weights,
            concat_bias,
            conv_bias_root->get_window_movement_strides(),
            conv_bias_root->get_window_dilation_strides(),
            conv_bias_root->get_padding_below(),
            conv_bias_root->get_padding_above(),
            conv_bias_root->get_data_dilation_strides(),
            conv_bias_root->with_relu());
        NGRAPH_DEBUG << "conv_horizontal_fusion: new cb shape "
                     << conv_bias_new->get_output_shape(0) << "\n";
        // slice
        size_t index = 0;
        for (auto cb : conv_bias_nodes)
        {
            auto slice_shape = cb->get_output_shape(0);
            NGRAPH_DEBUG << "conv_horizontal_fusion: slice shape " << slice_shape << "\n";
            auto lower_bounds = Coordinate{0, index, 0, 0};
            index += slice_shape[1];
            auto upper_bounds = Coordinate{slice_shape[0], index, slice_shape[2], slice_shape[3]};
            NGRAPH_DEBUG << "conv_horizontal_fusion: lower_bounds " << lower_bounds << "\n";
            NGRAPH_DEBUG << "conv_horizontal_fusion: upper_bounds " << upper_bounds << "\n";
            auto slice =
                std::make_shared<ngraph::op::Slice>(conv_bias_new, lower_bounds, upper_bounds);
            ngraph::replace_node(cb, slice);
        }

        return true;
    };

    auto m =
        make_shared<pattern::Matcher>(conv_bias, "CPUHorizontalFusion.CpuConvHorizontalFusion");
    this->add_matcher(m, callback);
}
