//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/pass/assign_placement.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/runtime/backend.hpp"

using namespace ngraph;
using namespace std;

pass::AssignPlacement::AssignPlacement(function<Placement(shared_ptr<Node>)> placement_policy)
    : m_placement_policy(placement_policy)
{
}

pass::AssignPlacement::AssignPlacement(vector<shared_ptr<runtime::Backend>> placement_backends)
    : m_placement_backends(placement_backends)
{
}

bool pass::AssignPlacement::run_on_node(shared_ptr<Node> node)
{
    if (!m_placement_backends.empty())
    {
        size_t backend_index = 0;
        for (auto backend : m_placement_backends)
        {
            backend_index += 1;
            if (backend->is_supported(*node))
            {
                node->set_placement(backend_index);

                return false;
            }
        }
    }
    else
    {
        node->set_placement(m_placement_policy(node));
    }

    return false;
}
