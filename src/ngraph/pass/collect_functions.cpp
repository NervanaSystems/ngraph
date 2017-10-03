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

#include "ngraph/ops/op.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/pass/collect_functions.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/util.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::pass;

bool CollectFunctions::run_on_function(ngraph::Function* func)
{
    traverse_nodes(func->get_result(), [&](Node* node)
    {
        if (typeid(ngraph::op::FunctionCall) == typeid(*node))
        {
            NGRAPH_INFO;
        }
    });

    return false;
}
