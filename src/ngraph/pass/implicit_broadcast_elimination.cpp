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

#include "ngraph/pass/implicit_broadcast_elimination.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/binary_elementwise_comparison.hpp"
#include "ngraph/op/util/binary_elementwise_logical.hpp"

using namespace std;
using namespace ngraph;

bool ngraph::pass::ImplicitBroadcastElimination::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    if (node->supports_auto_broadcast())
    {
        if (node->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            auto new_args = pass::explicit_broadcast(node);
            for (size_t i = 0; i < new_args.size(); i++)
            {
                node->input(i).replace_source_output(new_args[i]->output(0));
            }
            return true;
        }
    }
    return false;
}
