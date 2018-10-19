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

#include "ngraph/runtime/interpreter/int_placement.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/runtime/backend.hpp"

using namespace std;
using namespace ngraph;

// The policy that place supported ops on interpreter
// There are 3 levels of op support
// - Fully supported: List in fully_supported_ops
// - Partially supported: Listed in the partially supported ops. Only run on nnp if specific
//   conditions are satisfied.
// - Unsupported: Not listed at all
Placement runtime::interpreter::default_placement_policy(const std::shared_ptr<Node>& node)
{
    NGRAPH_INFO << "runtime::interpreter::default_placement_policy -Begin " + node->description();

    NGRAPH_INFO
        << "runtime::interpreter::default_placement_policy -End & placement on INTERPRETER ";
    // All ops by default are supported on interpreter
    // this is nGraph feature
    return Placement::INTERPRETER;
}
