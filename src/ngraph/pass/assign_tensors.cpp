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

#include "assign_tensors.hpp"

#include <exception>
#include <sstream>

#include "log.hpp"
#include "ngraph/ngraph.hpp"
#include "propagate_types.hpp"

using namespace std;
using namespace ngraph;

bool pass::AssignTensors::run_on_call_list(std::list<Node*>& node_list)
{
    for (Node* node : node_list)
    {
        try
        {
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

void pass::AssignTensors::check_dependencies(
    const std::vector<std::shared_ptr<CallBase>>& registered_passes) const
{
    bool found_propagate_types = false;
    for (auto pass : registered_passes)
    {
        if (dynamic_pointer_cast<PropagateTypes>(pass))
        {
            found_propagate_types = true;
        }
    }

    if (!found_propagate_types)
    {
        throw runtime_error("Depencency 'PropagateTypes' not found for pass 'AssignTensors'");
    }
}
