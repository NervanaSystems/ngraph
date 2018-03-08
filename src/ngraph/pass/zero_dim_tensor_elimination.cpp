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

#include "zero_dim_tensor_elimination.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/nop.hpp"
#include "ngraph/ops/sum.hpp"

using namespace ngraph;

static bool has_zero_dim(std::shared_ptr<Node> node)
{
    if (node->get_output_size() != 1)
    {
        throw ngraph_error("has_zero_dim is called on multi-output op");
    }
    return shape_size(node->get_outputs().at(0).get_shape()) == 0;
}

bool ngraph::pass::ZeroDimTensorElimination::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        //don't try to replace `op::Result` or (TODO:) multiple-output nodes
        if (n->is_output() || n->get_outputs().size() > 1)
        {
            continue;
        }

        //
        if (has_zero_dim(n))
        {
            auto nop = std::make_shared<op::Nop>(n->get_element_type(), n->get_shape());
            replace_node(n, nop);
            replaced = true;
            continue;
        }

        if (n->get_inputs().size() != 1 || !std::dynamic_pointer_cast<op::Nop>(n->get_input_op(0)))
        {
            continue;
        }

        if (auto sum = std::dynamic_pointer_cast<op::Sum>(n))
        {
            auto cvals = std::vector<std::string>(shape_size(n->get_shape()), std::string("0"));
            auto constant =
                std::make_shared<op::Constant>(n->get_element_type(), n->get_shape(), cvals);
            replaced = true;
            replace_node(n, constant);
        }
    }

    return replaced;
}
