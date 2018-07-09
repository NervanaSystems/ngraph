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

#include <algorithm>
#include <cassert>
#include <exception>
#include <map>
#include <sstream>
#include <unordered_set>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static void get_constant_tensors(const vector<Node*>& ops,
                                 unordered_set<descriptor::Tensor*>& tensors)
{
    for (const Node* n : ops)
    {
        if (auto constant_node = dynamic_cast<const op::Constant*>(n))
        {
            for (size_t i = 0; i < constant_node->get_output_size(); ++i)
            {
                descriptor::Tensor& t = constant_node->get_output_tensor(i);
                tensors.insert(&t);
            }
        }
    }
}

static void get_parameter_tensors(const ngraph::Function& f,
                                  unordered_set<descriptor::Tensor*>& tensors)
{
    for (const shared_ptr<op::Parameter>& n : f.get_parameters())
    {
        for (size_t i = 0; i < n->get_output_size(); ++i)
        {
            descriptor::Tensor& t = n->get_output_tensor(i);
            tensors.insert(&t);
        }
    }
}

static void get_result_tensors(const ngraph::Function& f,
                               unordered_set<descriptor::Tensor*>& tensors)
{
    for (const shared_ptr<op::Result>& n : f.get_results())
    {
        for (size_t i = 0; i < n->get_output_size(); ++i)
        {
            descriptor::Tensor& t = n->get_output_tensor(i);
            tensors.insert(&t);
        }
    }
}

using OpsSchedule = vector<Node*>;

static OpsSchedule get_ops_schedule(ngraph::Function& f)
{
    OpsSchedule ops_schedule;

    const list<shared_ptr<Node>> ops = f.get_ordered_ops();
    ops_schedule.reserve(ops.size());
    for (const shared_ptr<Node>& n : ops)
    {
        ops_schedule.push_back(n.get());
    }

    return ops_schedule;
}

bool pass::Liveness::run_on_function(shared_ptr<ngraph::Function> function)
{
    assert(function);
    ngraph::Function& f = *(function.get());

    // TODO: this should be computed just once and passed in to the function...
    const OpsSchedule ops_schedule = get_ops_schedule(f);

    unordered_set<descriptor::Tensor*> persistent_tensors;
    get_parameter_tensors(f, persistent_tensors);
    get_result_tensors(f, persistent_tensors);
    get_constant_tensors(ops_schedule, persistent_tensors);

    // For each tensor, this gives the number of uses associated with ops that have not yet been
    // visited during our chronological-order traversal of the schedule.
    map<descriptor::Tensor*, size_t> tensor_refcounts;

    for (Node* n : ops_schedule)
    {
        //----------------------------------------------------------------------------------------------
        // Identify and handle tensors that are no longer live...
        //----------------------------------------------------------------------------------------------
        n->liveness_free_list.clear();

        for (descriptor::Input& input_decl : n->get_inputs())
        {
            descriptor::Tensor* t = &(input_decl.get_tensor());

            if (persistent_tensors.find(t) != persistent_tensors.end())
            {
                continue;
            }

            const auto iter = tensor_refcounts.find(t);
            assert(iter != tensor_refcounts.end());

            size_t& refcount = iter->second;
            assert(refcount > 0);
            --refcount;

            if (refcount == 0)
            {
                n->liveness_free_list.insert(t);
                tensor_refcounts.erase(iter);
            }
        }

        //----------------------------------------------------------------------------------------------
        // Identify and handle newly created tensors...
        //----------------------------------------------------------------------------------------------
        n->liveness_new_list.clear();

        for (size_t i = 0; i < n->get_output_size(); ++i)
        {
            descriptor::Tensor* t = &(n->get_output_tensor(i));

            if (persistent_tensors.find(t) != persistent_tensors.end())
            {
                continue;
            }

            const size_t num_uses = n->get_output_inputs(i).size();
            tensor_refcounts[t] = num_uses;
            n->liveness_new_list.insert(t);

            if (num_uses == 0)
            {
                // A tensor with no uses is (apparently) supposed to appear in bot the 'liveness_new_list'
                // *and* the 'liveness_free_list' of the node that creates it.
                n->liveness_free_list.insert(t);
            }
        }
    }

    return false;
}
