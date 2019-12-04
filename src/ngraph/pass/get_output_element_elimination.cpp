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

#include <set>

#include "get_output_element_elimination.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/get_output_element.hpp"

using namespace ngraph;
using namespace std;

bool pass::GetOutputElementElimination::run_on_function(shared_ptr<Function> f)
{
    bool optimized = false;
    for (auto& n : f->get_ops())
    {
        for (auto& input : n->inputs())
        {
            if (auto goe = as_type<op::GetOutputElement>(input.get_source_output().get_node()))
            {
                input.replace_source_output(goe->input(0).get_source_output());
                // we don't need to fix anything w.r.t GetOutputElement as it will become
                // unreachable
                optimized = true;
            }
        }
    }
    return optimized;
}
