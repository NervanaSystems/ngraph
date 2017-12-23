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

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::descriptor;

bool pass::Liveness::run_on_call_graph(list<shared_ptr<Node>>& ops)
{
    unordered_set<Tensor*> currently_live;

    for (auto it = ops.rbegin(); it != ops.rend(); it++)
    {
        shared_ptr<Node> node = *it;
        node->liveness_live_list.clear();
        node->liveness_new_list.clear();
        node->liveness_free_list.clear();
        unordered_set<Tensor*> input_tensor_decls;
        for (Input& input_decl : node->get_inputs())
        {
            Tensor& tensor = input_decl.get_tensor();
            if (is_temporary(tensor))
            {
                input_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<Tensor*> output_tensor_decls;
        for (size_t i = 0; i < node->get_num_outputs(); ++i)
        {
            Tensor& tensor = node->get_output_tensor(i);
            if (is_temporary(tensor))
            {
                output_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<Tensor*> free_tensor_decls;
        unordered_set<Tensor*> new_tensor_decls;
        unordered_set<Tensor*> all_tensor_decls = input_tensor_decls;
        all_tensor_decls.insert(output_tensor_decls.begin(), output_tensor_decls.end());

        for (Tensor* tensor_decl : all_tensor_decls)
        {
            if (!contains(currently_live, tensor_decl))
            {
                // this is the last node that value is seen in
                // delete it at the end of the op
                currently_live.insert(tensor_decl);
                free_tensor_decls.insert(tensor_decl);
            }
        }

        node->liveness_live_list = currently_live;
        for (Tensor* output_decl : output_tensor_decls)
        {
            if (contains(currently_live, output_decl))
            {
                new_tensor_decls.insert(output_decl);
                currently_live.erase(output_decl);
            }
        }
        node->liveness_free_list = free_tensor_decls;
        node->liveness_new_list = new_tensor_decls;
    }

    // Anything marked as output must remain live for the remainder of the graph
    // Add outputs to live_list and remove from free_list
    unordered_set<Tensor*> outputs;
    unordered_set<Tensor*> seen;
    for (shared_ptr<Node> node : ops)
    {
        for (Tensor* tensor : node->liveness_live_list)
        {
            if (tensor->is_output())
            {
                outputs.insert(tensor);
            }
        }
        for (Tensor* tensor : outputs)
        {
            node->liveness_live_list.insert(tensor);
            node->liveness_free_list.erase(tensor);

            if (contains(node->liveness_new_list, tensor))
            {
                if (contains(seen, tensor))
                {
                    node->liveness_new_list.erase(tensor);
                }
                else
                {
                    seen.insert(tensor);
                }
            }
        }
    }

    // validate_liveness(ops);
    return false;
}

bool pass::Liveness::is_temporary(const Tensor& tensor)
{
    return tensor.is_persistent() == false && tensor.is_input() == false &&
           tensor.is_output() == false && tensor.is_constant() == false;
    // && tensor.is_compile_only() == false;
}

void pass::Liveness::validate_liveness(const list<Node*>& ops)
{
    unordered_set<Tensor*> dead_tensors;
    for (const Node* node : ops)
    {
        auto active = node->liveness_live_list;
        active.insert(node->liveness_new_list.begin(), node->liveness_new_list.end());
        active.insert(node->liveness_free_list.begin(), node->liveness_free_list.end());
        for (const Tensor* tensor : active)
        {
            if (contains(dead_tensors, tensor))
            {
                throw runtime_error("Liveness: Dead tensors intersect active tensors");
            }
        }
        dead_tensors.insert(node->liveness_free_list.begin(), node->liveness_free_list.end());
    }
}
