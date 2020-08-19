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

#include "cpu_workspace_insertion.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/conv_fused.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"

using namespace ngraph;

static std::shared_ptr<pattern::Matcher> create_maxpool_with_indices_matcher()
{
    Shape shape_data{1, 1, 14};
    auto data = std::make_shared<pattern::op::Label>(element::f32, shape_data);
    Shape window_shape{3};
    auto max_pool = std::make_shared<op::v0::MaxPool>(data, window_shape);
    auto delta = std::make_shared<pattern::op::Label>(element::f32, max_pool->get_output_shape(0));
    auto is_max_pool = pattern::has_class<op::v0::MaxPool>();
    auto max_pool_label = std::make_shared<pattern::op::Label>(
        element::f32, max_pool->get_output_shape(0), is_max_pool);
    auto max_pool_bprop =
        std::make_shared<op::v0::MaxPoolBackprop>(data,
                                                  delta,
                                                  max_pool_label,
                                                  max_pool->get_window_shape(),
                                                  max_pool->get_window_movement_strides(),
                                                  max_pool->get_padding_below(),
                                                  max_pool->get_padding_above());
    return std::make_shared<pattern::Matcher>(max_pool_bprop);
}

bool runtime::cpu::pass::CPUWorkspaceInsertion::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    auto matcher = create_maxpool_with_indices_matcher();

    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (n->is_output() || n->is_parameter())
        {
            continue;
        }

        if (matcher->match(n) && transform(*matcher))
        {
            replaced = true;
        }
    }

    return replaced;
}

bool runtime::cpu::pass::CPUWorkspaceInsertion::transform(pattern::Matcher& m)
{
    auto data = std::static_pointer_cast<pattern::op::Label>(
        m.get_pattern_value().get_node()->get_argument(0));
    auto delta = std::static_pointer_cast<pattern::op::Label>(
        m.get_pattern_value().get_node()->get_argument(1));
    auto max_pool = std::static_pointer_cast<pattern::op::Label>(
        m.get_pattern_value().get_node()->get_argument(2));
    NGRAPH_DEBUG << "In a callback for construct_max_pool_with_indices against "
                 << m.get_match_root()->get_name();

    auto pattern_map = m.get_pattern_map();
    auto m_max_pool = std::static_pointer_cast<op::v0::MaxPool>(pattern_map[max_pool]);
    auto m_max_pool_bprop = m.get_match_root_as<op::v0::MaxPoolBackprop>();
    NGRAPH_CHECK(m_max_pool_bprop,
                 "match root node ",
                 *m.get_match_root(),
                 " not of type `op::v0::MaxPoolBackprop`");

    if (m_max_pool_bprop->get_output_shape(0).size() != 4 ||
        m_max_pool_bprop->get_window_shape().size() != 2 ||
        m_max_pool_bprop->get_input_element_type(0) != element::f32)
    {
        NGRAPH_DEBUG << "DNNL doesn't support inputs of given shape type";
        return false;
    }

    auto max_pool_with_indices =
        std::make_shared<op::MaxPoolWithIndices>(pattern_map[data],
                                                 m_max_pool->get_window_shape(),
                                                 m_max_pool->get_window_movement_strides(),
                                                 m_max_pool->get_padding_below(),
                                                 m_max_pool->get_padding_above());

    // rewire users to use a new MaxPoolWithIndices (maxpool's output)
    for (Output<Node> o : m_max_pool->outputs())
    {
        std::set<Input<Node>> copy = o.get_target_inputs();
        for (Input<Node> i : copy)
        {
            i.replace_source_output(max_pool_with_indices->output(0));
        }
    }

    // create a new max_pool_with_indices_bprop
    auto max_pool_with_indices_bprop =
        std::make_shared<op::MaxPoolWithIndicesBackprop>(pattern_map[data],
                                                         pattern_map[delta],
                                                         max_pool_with_indices->output(1),
                                                         m_max_pool->get_window_shape(),
                                                         m_max_pool->get_window_movement_strides(),
                                                         m_max_pool->get_padding_below(),
                                                         m_max_pool->get_padding_above());

    ngraph::replace_node(m_max_pool_bprop, max_pool_with_indices_bprop);
    if (m_return_indices)
    {
        m_index_list.push_back(max_pool_with_indices->output(1));
    }
    return true;
}
