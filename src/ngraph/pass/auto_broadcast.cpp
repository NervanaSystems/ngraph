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

#include "ngraph/pass/auto_broadcast.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

using namespace std;
using namespace ngraph;

bool ngraph::pass::AutoBroadcast::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    bool modified = false;

    if (auto op = std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(node))
    {
        if (op->get_autob() != op::AutoBcastType::NONE)
        {
            auto new_args = op->auto_broadcast();
            size_t i = 0;
            for (size_t i = 0; i < new_args.size(); i++)
            {
                op->input(i).replace_source_output(new_args[i]->output(0));
            }
            modified = true;
        };
    }

    return modified;
}
