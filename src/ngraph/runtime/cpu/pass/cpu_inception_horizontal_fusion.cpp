/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/cpu/pass/cpu_inception_horizontal_fusion.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"

using namespace ngraph;
using namespace std;

void ngraph::runtime::cpu::pass::CPUInceptionHorizontalFusion::cpu_inception_horizontal_fusion()
{
    auto is_concat_or_maxpool = [](std::shared_ptr<Node> n) {
        return std::dynamic_pointer_cast<ngraph::op::Concat>(n) ||
               std::dynamic_pointer_cast<ngraph::op::MaxPool>(n);
    };

    auto data_conv = std::make_shared<pattern::op::Label>(
        element::f32, Shape{1, 256, 35, 35}, is_concat_or_maxpool);
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

    pattern::graph_rewrite_callback callback = [data_conv](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for inception horizontal fusion for "
                     << m.get_match_root()->get_name();

        auto conv_bias = std::dynamic_pointer_cast<op::ConvolutionBias>(m.get_match_root());

        //check if the node has been replaced
        if (conv_bias->get_users().empty())
        {
            return false;
        }

        //check if 1x1 filter
        auto m_filters_shape = conv_bias->get_input_shape(1);
        if (m_filters_shape[2] != 1 && m_filters_shape[3] != 1)
        {
            NGRAPH_DEBUG << "inception: not 1x1 filter\n";
            return false;
        }

        // get weights and bias from each CBR 1x1 and create Concat nodes
        std::vector<std::shared_ptr<Node>> weights_nodes;
        std::vector<std::shared_ptr<Node>> bias_nodes;
        std::vector<std::shared_ptr<Node>> conv_bias_nodes;

        for (auto u : m.get_pattern_map()[data_conv]->get_users())
        {
            if (!pattern::has_class<ngraph::op::ConvolutionBias>()(u))
            {
            	continue;
            }
            auto u_filters_shape = u->get_input_shape(1);
            if (u_filters_shape[2] != 1 || u_filters_shape[3] != 1)
            {
                return false;
            }
            weights_nodes.push_back(u->get_argument(1));
            bias_nodes.push_back(u->get_argument(2));
            conv_bias_nodes.push_back(u);
        }
        if (conv_bias_nodes.size() == 1)
        {
            return false;
        }
        auto concat_weights = std::make_shared<ngraph::op::Concat>(weights_nodes, 0);
        auto concat_bias = std::make_shared<ngraph::op::Concat>(bias_nodes, 0);
        auto conv_bias_new =
            std::make_shared<ngraph::op::ConvolutionBias>(conv_bias->get_argument(0),
                                                          concat_weights,
                                                          concat_bias,
                                                          conv_bias->get_window_movement_strides(),
                                                          conv_bias->get_window_dilation_strides(),
                                                          conv_bias->get_padding_below(),
                                                          conv_bias->get_padding_above(),
                                                          conv_bias->get_data_dilation_strides(),
                                                          conv_bias->with_relu());
        NGRAPH_DEBUG << "inception: new cb shape " << conv_bias_new->get_output_shape(0) << "\n";
        //slice
        size_t index = 0;
        for (auto cb : conv_bias_nodes)
        {
            auto slice_shape = cb->get_output_shape(0);
            NGRAPH_DEBUG << "inception: slice shape " << slice_shape << "\n";
            auto lower_bounds = Coordinate{0, index, 0, 0};
            index += slice_shape[1];
            auto upper_bounds = Coordinate{slice_shape[0], index, slice_shape[2], slice_shape[2]};
            NGRAPH_DEBUG << "inception: lower_bounds " << lower_bounds << "\n";
            NGRAPH_DEBUG << "inception: upper_bounds " << upper_bounds << "\n";
            auto slice =
                std::make_shared<ngraph::op::Slice>(conv_bias_new, lower_bounds, upper_bounds);
            ngraph::replace_node(cb, slice);
        }

        return true;
    };

    auto m = make_shared<pattern::Matcher>(conv_bias, callback);
    this->add_matcher(m);
}
