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

#include <exception>
#include <sstream>
#include <unordered_set>

#include "ngraph/log.hpp"
#include "ngraph/function.hpp"
#include "ngraph/runtime/mlir/pass/ngraph_dialect_lower.hpp"
#include "ngraph/runtime/mlir/mlir_ngraph_ops.hpp"

using namespace std;
using namespace ngraph;

bool pass::NgraphDialectLower::run_on_function(shared_ptr<Function> function)
{
    for (shared_ptr<Node> op : function->get_ordered_ops())
    {
        NGRAPH_INFO << *op;
        switch(runtime::mlir::get_typeid(*op))
        {
            case runtime::mlir::OP_TYPEID::Add_v0:
            NGRAPH_INFO << "Add";
            break;
        }
    }

    return false;
}
