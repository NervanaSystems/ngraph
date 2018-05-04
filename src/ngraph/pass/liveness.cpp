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

#include <exception>
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

bool pass::Liveness::run_on_function(shared_ptr<ngraph::Function> function)
{
    list<shared_ptr<Node>> ops = function->get_ordered_ops();

    unordered_set<descriptor::Tensor*> persistent_tensors;
    unordered_set<descriptor::Tensor*> output_tensors;
    for (shared_ptr<op::Parameter> node : function->get_parameters())
    {
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            descriptor::Tensor& tensor = node->get_output_tensor(i);
            persistent_tensors.insert(&tensor);
        }
    }
    for (shared_ptr<op::Result> node : function->get_results())
    {
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            descriptor::Tensor& tensor = node->get_output_tensor(i);
            persistent_tensors.insert(&tensor);
            output_tensors.insert(&tensor);
        }
    }
    for (shared_ptr<Node> node : function->get_ordered_ops())
    {
        if (auto constant_node = dynamic_pointer_cast<op::Constant>(node))
        {
            for (size_t i = 0; i < constant_node->get_output_size(); ++i)
            {
                descriptor::Tensor& tensor = constant_node->get_output_tensor(i);
                persistent_tensors.insert(&tensor);
            }
        }
    }

    unordered_set<descriptor::Tensor*> currently_live;
    for (auto it = ops.rbegin(); it != ops.rend(); it++)
    {
        shared_ptr<Node> node = *it;
        node->liveness_live_list.clear();
        node->liveness_new_list.clear();
        node->liveness_free_list.clear();
        unordered_set<descriptor::Tensor*> input_tensor_decls;
        for (descriptor::Input& input_decl : node->get_inputs())
        {
            descriptor::Tensor& tensor = input_decl.get_tensor();
            if (!contains(persistent_tensors, &tensor))
            {
                input_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> output_tensor_decls;
        for (size_t i = 0; i < node->get_output_size(); ++i)
        {
            descriptor::Tensor& tensor = node->get_output_tensor(i);
            if (!contains(persistent_tensors, &tensor))
            {
                output_tensor_decls.insert(&tensor);
            }
        }

        unordered_set<descriptor::Tensor*> free_tensor_decls;
        unordered_set<descriptor::Tensor*> new_tensor_decls;
        unordered_set<descriptor::Tensor*> all_tensor_decls = input_tensor_decls;
        all_tensor_decls.insert(output_tensor_decls.begin(), output_tensor_decls.end());

        for (descriptor::Tensor* tensor_decl : all_tensor_decls)
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
        for (descriptor::Tensor* output_decl : output_tensor_decls)
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
    unordered_set<descriptor::Tensor*> outputs;
    unordered_set<descriptor::Tensor*> seen;
    for (shared_ptr<Node> node : ops)
    {
        for (descriptor::Tensor* tensor : node->liveness_live_list)
        {
            if (contains(output_tensors, tensor))
            {
                outputs.insert(tensor);
            }
        }
        for (descriptor::Tensor* tensor : outputs)
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

void pass::Liveness::validate_liveness(const list<Node*>& ops)
{
    unordered_set<descriptor::Tensor*> dead_tensors;
    for (const Node* node : ops)
    {
        auto active = node->liveness_live_list;
        active.insert(node->liveness_new_list.begin(), node->liveness_new_list.end());
        active.insert(node->liveness_free_list.begin(), node->liveness_free_list.end());
        for (const descriptor::Tensor* tensor : active)
        {
            if (contains(dead_tensors, tensor))
            {
                throw runtime_error("Liveness: Dead tensors intersect active tensors");
            }
        }
        dead_tensors.insert(node->liveness_free_list.begin(), node->liveness_free_list.end());
    }
}
