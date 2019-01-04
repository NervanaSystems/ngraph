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

#include "ngraph/runtime/hybrid/pass/fix_get_output_element.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/runtime/backend.hpp"

using namespace ngraph;
using namespace std;

runtime::hybrid::pass::FixGetOutputElement::FixGetOutputElement()
{
}

bool runtime::hybrid::pass::FixGetOutputElement::run_on_node(shared_ptr<Node> node)
{
    if (node->description() == "GetOutputElement")
    {
        auto parent = node->get_arguments().at(0);
        node->set_placement_index(parent->get_placement_index());
    }
    return false;
}
