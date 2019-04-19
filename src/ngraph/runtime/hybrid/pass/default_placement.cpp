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

#include "ngraph/runtime/hybrid/pass/default_placement.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/runtime/backend.hpp"

using namespace ngraph;
using namespace std;

runtime::hybrid::pass::DefaultPlacement::DefaultPlacement(
    const vector<shared_ptr<runtime::Backend>>& placement_backends)
    : m_placement_backends(placement_backends)
{
}

bool runtime::hybrid::pass::DefaultPlacement::run_on_node(shared_ptr<Node> node)
{
    size_t backend_index = 0;
    for (auto backend : m_placement_backends)
    {
        if (backend->is_supported(*node))
        {
            node->set_placement_index(backend_index);
            return false;
        }
        backend_index++;
    }
    throw runtime_error("Node " + node->get_name() + " not supported by any backend");
}
