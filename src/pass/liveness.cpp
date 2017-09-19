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

#include <exception>
#include <sstream>
#include <unordered_set>

#include "log.hpp"
#include "ngraph.hpp"
#include "pass/assign_tensors.hpp"
#include "pass/liveness.hpp"
#include "util.hpp"
#include "log.hpp"

using namespace std;
using namespace ngraph;

bool pass::Liveness::run_on_call_list(list<Node*>& ops)
{
    unordered_set<descriptor::Tensor*> currently_live;

    for(auto it=ops.rbegin(); it!=ops.rend(); it++)
    {
        Node& exop = **it;
        exop.liveness_live_list.clear();
        exop.liveness_new_list.clear();
        exop.liveness_free_list.clear();
        unordered_set<descriptor::Tensor*> input_tensor_decls;
        for (auto input_decl : exop.get_inputs())
        {
            descriptor::Tensor& tensor = input_decl.get_tensor();
            if (is_temporary(tensor))
            {
                input_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> output_tensor_decls;
        for (auto output_decl : exop.get_outputs())
        {
            descriptor::Tensor& tensor = output_decl.get_tensor();
            if (is_temporary(tensor))
            {
                output_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> free_tensor_decls;
        unordered_set<descriptor::Tensor*> new_tensor_decls;
        unordered_set<descriptor::Tensor*> all_tensor_decls = input_tensor_decls;

        for (auto decls : {input_tensor_decls, output_tensor_decls})
        {
            for (descriptor::Tensor* tensor_decl : decls)
            {
                if (!contains(currently_live, tensor_decl))
                {
                    // this is the last node that value is seen in
                    // delete it at the end of the op
                    currently_live.insert(tensor_decl);
                    free_tensor_decls.insert(tensor_decl);
                }
            }
        }

        exop.liveness_live_list = currently_live;
        for (descriptor::Tensor* output_decl : output_tensor_decls)
        {
            if (contains(currently_live, output_decl))
            {
                new_tensor_decls.insert(output_decl);
                currently_live.erase(output_decl);
            }
        }
        exop.liveness_free_list = free_tensor_decls;
        exop.liveness_new_list = new_tensor_decls;
    }

    // Anything marked as output must remain live for the remainder of the graph
    // Add outputs to live_list and remove from free_list
    unordered_set<descriptor::Tensor*> outputs;
    unordered_set<descriptor::Tensor*> seen;
    for (Node* exop : ops)
    {
        for (descriptor::Tensor* tensor : exop->liveness_live_list)
        {
            if (tensor->is_output())
            {
                outputs.insert(tensor);
            }
        }
        for (descriptor::Tensor* tensor : outputs)
        {
            exop->liveness_live_list.insert(tensor);
            exop->liveness_free_list.erase(tensor);

            if (contains(exop->liveness_new_list, tensor))
            {
                if (contains(seen, tensor))
                {
                    exop->liveness_new_list.erase(tensor);
                }
                else
                {
                    seen.insert(tensor);
                }
            }
        }
    }

    validate_liveness(ops);
    return false;
}

void pass::Liveness::check_dependencies(
    const std::vector<std::shared_ptr<CallBase>>& registered_passes) const
{
    bool found_propagate_types = false;
    for (auto pass : registered_passes)
    {
        if (dynamic_pointer_cast<AssignTensors>(pass))
        {
            found_propagate_types = true;
        }
    }

    if (!found_propagate_types)
    {
        throw runtime_error("Dependency 'PropagateTypes' not found for pass 'AssignTensors'");
    }
}

bool pass::Liveness::is_temporary(const descriptor::Tensor& tensor)
{
    return
        tensor.is_persistent() == false
        && tensor.is_input() == false
        ;
        // && tensor.is_constant() == false
        // && tensor.is_compile_only() == false;
}

void pass::Liveness::validate_liveness(const list<Node*>& ops)
{
    unordered_set<descriptor::Tensor*> dead_tensors;
    for (const Node* exop : ops)
    {
        auto active = exop->liveness_live_list;
        active.insert(exop->liveness_new_list.begin(), exop->liveness_new_list.end());
        active.insert(exop->liveness_free_list.begin(), exop->liveness_free_list.end());
        for (const descriptor::Tensor* tensor : active)
        {
            if (contains(dead_tensors, tensor))
            {
                throw runtime_error("Liveness: Dead tensors intersect active tensors");
            }
        }
        dead_tensors.insert(exop->liveness_free_list.begin(), exop->liveness_free_list.end());
    }
}

