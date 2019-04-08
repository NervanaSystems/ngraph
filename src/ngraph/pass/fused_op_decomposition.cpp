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
#include "ngraph/op/util/fused_op.hpp"

using namespace std;
using namespace ngraph;

bool ngraph::pass::FusedOpDecomposition::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    bool modified = false;

    if (auto fused_op = std::dynamic_pointer_cast<ngraph::op::util::FusedOp>(node))
    {
        auto subgraph = fused_op->decompose_op();
        if (subgraph.size() != fused_op->get_output_size())
        {
            throw ngraph_error("While replacing " + node->get_name() +
                               ", mismatch between op output count and outputs of the decomposed "
                               "subgraph. Expected: " +
                               to_string(fused_op->get_output_size()) + " Got: " +
                               to_string(subgraph.size()));
        }
        if (fused_op->get_output_size() == 1)
        {
            ngraph::replace_node(fused_op, subgraph[0]);
        }
        else
        {
            // TODO (jbobba): Handle multi-output ops. Need to find the GOE for the output and replace that with subgraph output node
        }
        modified = true;
    }

    return modified;
}
