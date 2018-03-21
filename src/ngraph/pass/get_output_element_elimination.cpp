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

#include <set>

#include "get_output_element_elimination.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sum.hpp"

using namespace ngraph;

bool ngraph::pass::GetOutputElementElimination::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool optimized = false;
    for (auto n : f->get_ordered_ops())
    {
        for (auto& input : n->get_inputs())
        {
            if (auto goe =
                    std::dynamic_pointer_cast<op::GetOutputElement>(input.get_output().get_node()))
            {
                auto multi = goe->get_inputs().at(0).get_output().get_node();
                input.replace_output(goe->get_inputs().at(goe->get_n()).get_output());

                //fix node arguments
                auto& n_args =
                    const_cast<ngraph::NodeVector&>(n->get_arguments_FOR_GRAPH_REWRITE_ONLY());
                auto it = std::find(begin(n_args), end(n_args), goe);
                if (it == end(n_args))
                {
                    throw ngraph_error("Expected to find GetOutputElement in n's inputs");
                }
                *it = multi;

                //fix multi's users
                const_cast<std::multiset<Node*>&>(multi->users()).insert(n.get());

                //we don't need to fix anything w.r.t GetOutputElement as it will become unreachable
                optimized = true;
            }
        }
    }
    return optimized;
}
