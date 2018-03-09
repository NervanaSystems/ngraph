/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <stack>

#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/parameter.hpp"
#include "cpu_rnn_mat_fusion.hpp"

static ngraph::NodeVector get_users(ngraph::Node& node)
{
    ngraph::NodeVector result;

    for (size_t i = 0; i < node.get_output_size(); ++i)
    {
        for (auto input : node.get_output_inputs(i))
        {
            result.push_back(input->get_node());
        }
    }

    return result;
}

#define TI(x) std::type_index(typeid(x))

void FindDot(std::shared_ptr<ngraph::Node> node, )
bool ngraph::runtime::cpu::pass::CPURnnMatFusion::run_on_function(
    std::shared_ptr<ngraph::Function> function)
{
    bool clobbered = false;
    std::cout << "Slice: " << TI(ngraph::op::Slice).hash_code() << std::endl;
    std::cout << "Reshape: " << TI(ngraph::op::Reshape).hash_code() << std::endl;
    std::cout << "Dot: " << TI(ngraph::op::Dot).hash_code() << std::endl;

    std::list<std::shared_ptr<Node>> param_nodes;
    for (auto& n : function->get_ordered_ops())
    {
        // Work around a warning [-Wpotentially-evaluated-expression]
        Node& node = *n;
        std::string type = "other";
        if (TI(node) == TI(ngraph::op::Parameter)) {
            param_nodes.push_back(n);
        }
        if (TI(node) == TI(ngraph::op::Slice)) {
            type = "Slice";
        }
        if (TI(node) == TI(ngraph::op::Reshape)) {
            type = "Reshape";
        }
        if (TI(node) == TI(ngraph::op::Dot)) {
            type = "Dot";
        }
        std::cout << "node (" << type << "): " << node.get_friendly_name() << std::endl;
        for (const auto& in : node.get_input_ops()) {
            std::cout << "    in:  " << in->get_friendly_name() << std::endl;
        }
        auto outputs = get_users(node);
        for (const auto& out : outputs) {
            std::cout << "    out: " << out->get_friendly_name() << std::endl;
        }
    }
    for (auto& p : param_nodes) {

    }

    return clobbered;
}
