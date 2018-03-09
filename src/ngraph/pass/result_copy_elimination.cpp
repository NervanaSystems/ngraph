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

#include "result_copy_elimination.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/result.hpp"
#include "ngraph/util.hpp"

bool ngraph::pass::ResultCopyElimination::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    std::set<std::shared_ptr<Node>> seen;
    for (auto res : f->get_results())
    {
        auto arg = res->get_input_op(0);
        //we need a copy
        if (arg->is_parameter() || arg->is_constant())
        {
            continue;
        }

        //TODO: check if broadcast replace op::Result w/ a copy of broadcast node

        //TODO: consider other cases where it's easier to recompute than make a copy

        //we will compute the result directly into output[]
        if (seen.count(arg) == 0)
        {
            res->set_needs_copy(false);
            seen.insert(arg);
        }
    }

    return true;
}
