// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/pass/assign_tensors.hpp"

#include <exception>
#include <sstream>

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"

using namespace std;
using namespace ngraph;

bool pass::AssignTensors::run_on_call_graph(list<std::shared_ptr<Node>>& nodes)
{
    for (shared_ptr<Node> node : nodes)
    {
        try
        {
            // We need to set the nodes is_output state prior to call assign_tensors
            // so that the output state can be passes to the constructed tensors.
            if (node == get_state().get_functions().at(0)->get_result())
            {
                node->set_is_output();
            }

            node->assign_tensors();
        }
        catch (exception& e)
        {
            stringstream ss;
            ss << "Error with node " << *node << ": ";
            ss << e.what();
            throw invalid_argument(ss.str());
        }
    }
    return false;
}
