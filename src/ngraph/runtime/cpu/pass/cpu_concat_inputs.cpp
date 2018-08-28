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
#include "cpu_concat_inputs.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <typeindex>
#include <typeinfo>
#include <unordered_set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"

using namespace ngraph;

void ngraph::runtime::cpu::pass::ConcatInputs::concat_lstm_inputs()
{
    auto ht_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});
    auto weights_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto xt = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});
    auto weights_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto bias1 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto bias2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});

    auto lstm = std::make_shared<op::Lstm>(xt, weights_i2h, ht_1, weights_h2h, bias1, bias2, ct_1);
    auto goe = std::make_shared<op::GetOutputElement>(lstm, 0);
    auto lstm_node_label = std::make_shared<pattern::op::Label>(goe, nullptr, NodeVector{goe});

    pattern::graph_rewrite_callback callback =
        [lstm_node_label, xt, weights_h2h, ht_1, weights_i2h, bias1, bias2, ct_1](
            pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_map();
            NGRAPH_DEBUG << " In LSTM MKLDNN callback";

            if (m.get_match_root()->get_element_type() != element::f32)
            {
                NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                             << " type is not float!";
                return false;
            }
            std::shared_ptr<Node> src_layer = pattern_map[xt];
            std::shared_ptr<Node> src_iter =
                std::make_shared<op::Concat>(NodeVector{pattern_map[ht_1], pattern_map[ct_1]}, 0);
            std::shared_ptr<Node> bias =
                std::make_shared<op::Add>(pattern_map[bias1], pattern_map[bias2]);

            auto lstm_node = pattern_map[lstm_node_label]->get_arguments()[0];
            auto batch_size = std::dynamic_pointer_cast<op::Lstm>(lstm_node)->get_batch_size();
            auto feature_size =
                std::dynamic_pointer_cast<op::Lstm>(lstm_node)->get_src_iter_feature_size();
            auto lstm_mkldnn_node = std::make_shared<op::Lstm>(
                src_layer, src_iter, pattern_map[weights_i2h], pattern_map[weights_h2h], bias);

            auto lstm_ht_out = std::make_shared<op::GetOutputElement>(lstm_mkldnn_node, 0);
            auto lstm_ht_ct_out = std::make_shared<op::GetOutputElement>(lstm_mkldnn_node, 1);

            // dst_iter of lstm mkldnn output holds the results of both recurrent state
            // tensor outputs. we need to slice the ct.
            auto ht_slice =
                std::make_shared<op::Slice>(lstm_ht_ct_out,
                                            Coordinate{0, 0},
                                            Coordinate{static_cast<unsigned long>(batch_size),
                                                       static_cast<unsigned long>(feature_size)});
            auto ct_slice =
                std::make_shared<op::Slice>(lstm_ht_ct_out,
                                            Coordinate{static_cast<unsigned long>(batch_size), 0},
                                            Coordinate{static_cast<unsigned long>(2 * batch_size),
                                                       static_cast<unsigned long>(feature_size)});

            // now go through the GOE'sand replace the slices(ht)
            std::set<std::shared_ptr<ngraph::Node>> lstm_outputs;
            for (auto& goes : lstm_node->get_outputs().at(0).get_inputs())
            {
                auto goe_node = std::dynamic_pointer_cast<op::GetOutputElement>(goes->get_node());
                lstm_outputs.insert(goes->get_node());
                // first output node of lstm
                if (goe_node->get_n() == 0)
                {
                    NGRAPH_DEBUG << "Replacing 1st output Lstm node " << goe_node->get_name()
                                 << " with " << lstm_ht_out->get_name();
                    if (!goe_node->get_users().empty())
                    {
                        ngraph::replace_node(goe_node, lstm_ht_out);
                    }
                }
                else if (goe_node->get_n() == 1)
                {
                    for (auto& goe_ct_user : goe_node->get_users())
                    {
                        for (size_t i = 0; i < goe_ct_user->get_input_size(); i++)
                        {
                            if (goe_ct_user->get_argument(i) == goe_node)
                            {
                                goe_ct_user->get_inputs().at(i).replace_output(
                                    ct_slice->get_outputs().at(0));
                            }
                        }
                    }
                    NGRAPH_DEBUG << "Replacing 2nd output Lstm node  " << goe_node->get_name()
                                 << " with " << ct_slice->get_name();
                }
            }

            if (lstm_outputs.find(m.get_match_root()) == lstm_outputs.end())
            {
                throw ngraph_error(
                    "Pattern matcher error, matched root node should be one of the LSTM outputs");
            }
            return true;
        };
    auto m = std::make_shared<pattern::Matcher>(lstm_node_label, callback);
    this->add_matcher(m);
}
