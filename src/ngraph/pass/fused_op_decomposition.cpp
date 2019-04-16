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

#include "ngraph/pass/fused_op_decomposition.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/util/fused_op.hpp"

using namespace std;
using namespace ngraph;

bool ngraph::pass::FusedOpDecomposition::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    bool modified = false;

    if (auto fused_op = std::dynamic_pointer_cast<ngraph::op::util::FusedOp>(node))
    {
        if (m_callback && m_callback(*node))
        {
            // Op supported by backend. Do not decompose
            return modified;
        }
        auto subgraph_outputs = fused_op->decompose_op();
        size_t i = 0;
        for (auto output_node : subgraph_outputs)
        {
            for (size_t j = 0; j < output_node->get_output_size(); j++, i++)
            {
                // TODO: Provenance
                auto target_inputs = fused_op->output(i).get_target_inputs();
                std::set<ngraph::Input<Node>> fop_users{begin(target_inputs), end(target_inputs)};
                for (auto fop_user : fop_users)
                {
                    if (auto goe = dynamic_cast<op::GetOutputElement*>(fop_user.get_node()))
                    {
                        if (goe->get_n() == i && !goe->output(0).get_target_inputs().empty())
                        {
                            // Replace GOE users
                            auto goe_target_inputs = goe->output(0).get_target_inputs();
                            std::set<ngraph::Input<Node>> goe_users{begin(goe_target_inputs),
                                                                    end(goe_target_inputs)};
                            for (auto goe_user : goe_users)
                            {
                                goe_user.replace_source_output(output_node->output(j));
                            }
                        }
                    }
                    else
                    {
                        fop_user.replace_source_output(output_node->output(j));
                    }
                }
            }
        }
        if (i != fused_op->get_output_size())
        {
            throw ngraph_error("While replacing " + node->get_name() +
                               ", mismatch between op output count and outputs of the decomposed "
                               "subgraph. Expected: " +
                               to_string(fused_op->get_output_size()) + " Got: " + to_string(i));
        }
        modified = true;
    }

    return modified;
}

pass::FusedOpDecomposition::FusedOpDecomposition(op_query_t callback)
    : m_callback{callback}
{
}
