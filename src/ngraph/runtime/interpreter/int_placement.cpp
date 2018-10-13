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

static bool all_ones(const vector<long unsigned int> v)
{
    for (auto elem : v)
    {
        if (elem != 1)
        {
            return false;
        }
    }
    return true;
}

static bool filter_shape_allowed(const Shape& shape)
{
    const size_t max_filter_dim = 16;
    for (size_t i = 2; i < shape.size(); i++)
    {
        if (shape[i] > max_filter_dim)
        {
            return false;
        }
    }
    return true;
}

// The policy that place supported ops on interpreter
// There are 3 levels of op support
// - Fully supported: List in fully_supported_ops
// - Partially supported: Listed in the partially supported ops. Only run on nnp if specific
//   conditions are satisfied.
// - Unsupported: Not listed at all
Placement runtime::interpreter::default_placement_policy(const std::shared_ptr<Node>& node)
{
    NGRAPH_INFO << "runtime::interpreter::default_placement_policy -Begin " + node->description();
    // clang-format off
    static unordered_set<string> fully_supported_ops = {
        "Abs",
        "Add", 
        "Parameter",
        "Result"
    };
    static unordered_set<string> partially_supported_ops = {
        "Dot"
    };
    // clang-format on

    string node_op = node->description();
    if (fully_supported_ops.count(node_op) == 0 && partially_supported_ops.count(node_op) == 0)
    {
        NGRAPH_INFO
            << "runtime::interpreter::default_placement_policy  placement done on CPU for " +
                   node->description();
        return Placement::CPU;
    }

    if (node_op == "Dot")
    {
        // Experimental
        if (shape_size(node->get_shape()) > 50000)
        {
            return Placement::CPU;
        }
    }

    NGRAPH_INFO
        << "runtime::interpreter::default_placement_policy -End & placement on INTERPRETER ";
    return Placement::INTERPRETER;
}
