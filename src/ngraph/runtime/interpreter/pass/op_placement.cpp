//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <numeric>

#include "ngraph/log.hpp"
#include "ngraph/runtime/interpreter/pass/op_placement.hpp"

using namespace ngraph;
using namespace std;

runtime::interpreter::pass::OpPlacement::OpPlacement(std::set<std::string> unsupported_ops)
    : m_unsupported_ops{unsupported_ops}
{
}

bool runtime::interpreter::pass::OpPlacement::run_on_function(
    std::shared_ptr<ngraph::Function> function)
{
    for (shared_ptr<Node> node : function->get_ops())
    {
        assign_placement(node);
    }
    return false;
}

void runtime::interpreter::pass::OpPlacement::assign_placement(shared_ptr<Node> node)
{
    if (is_supported_on_device(node) != DeviceSupport::SUPPORTED)
    {
        node->set_placement(1);
    }
    else
    {
        node->set_placement(0);
    }
}

runtime::interpreter::pass::OpPlacement::DeviceSupport
    runtime::interpreter::pass::OpPlacement::is_supported_on_device(shared_ptr<Node> node)
{
    DeviceSupport rc = DeviceSupport::UNKNOWN;
    NGRAPH_INFO << *node;
    return rc;
}
