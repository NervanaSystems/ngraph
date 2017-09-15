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

#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/runtime/eigen/add.hpp"
#include "ngraph/runtime/eigen/external_function.hpp"
#include "ngraph/runtime/eigen/multiply.hpp"
#include "ngraph/runtime/eigen/return.hpp"

using namespace std;
using namespace ngraph::runtime::eigen;

ExternalFunction::ExternalFunction()
    : m_instructions(make_shared<std::vector<std::shared_ptr<ngraph::runtime::Instruction>>>())
{
}

// Define code generators for handled ops.
std::unordered_map<std::type_index,
                   std::function<void(ngraph::Node*,
                                      ExternalFunction*,
                                      const std::vector<size_t>& inputs,
                                      const std::vector<size_t>& outputs)>>&
    ExternalFunction::get_op_map()
{
    static bool initialized = false;
    static std::unordered_map<std::type_index,
                              std::function<void(Node*,
                                                 ExternalFunction*,
                                                 const std::vector<size_t>& inputs,
                                                 const std::vector<size_t>& outputs)>>
        op_map;
    if (!initialized)
    {
        op_map[type_index(typeid(op::Add))] = [](Node*                      n,
                                                 ExternalFunction*          ef,
                                                 const std::vector<size_t>& in,
                                                 const std::vector<size_t>& out) {
            ef->get_instructions()->push_back(
                make_shared<runtime::eigen::AddInstruction<element::Float32>>(
                    in[0], in[1], out[0]));
        };

        op_map[type_index(typeid(op::Multiply))] = [](Node*                      n,
                                                      ExternalFunction*          ef,
                                                      const std::vector<size_t>& in,
                                                      const std::vector<size_t>& out) {
            ef->get_instructions()->push_back(
                make_shared<runtime::eigen::MultiplyInstruction<element::Float32>>(
                    in[0], in[1], out[0]));
        };

        op_map[type_index(typeid(op::Parameter))] = [](Node*                      n,
                                                       ExternalFunction*          ef,
                                                       const std::vector<size_t>& in,
                                                       const std::vector<size_t>& out) {};

        initialized = true;
    }
    return op_map;
}

void ExternalFunction::compile(std::shared_ptr<ngraph::Function> f)
{
    // This will be replaced with the pass manager
    // Get the ordered list of ops in execution order
    pass::TopologicalSort ts;
    ts.run_on_tree(f->get_result());
    auto nodes = ts.get_call_graph();
    // Types
    for (auto node : nodes)
    {
        node->propagate_types();
    }
    // Determine tensors
    for (auto node : nodes)
    {
        node->assign_tensors();
    }

    // Determine tensor requirements for  the call frame
    unordered_map<shared_ptr<ngraph::descriptor::TensorView>, size_t> tensor_index;
    // First come the function inputs
    for (auto param : f->get_parameters())
    {
        for (auto output : param->get_outputs())
        {
            auto   tv        = output.get_tensor_view();
            size_t index     = tensor_index.size();
            tensor_index[tv] = index;
        }
    }
    m_n_inputs = tensor_index.size();

    // Next are the function outputs
    for (auto output : f->get_result()->get_outputs())
    {
        auto   tv        = output.get_tensor_view();
        size_t index     = tensor_index.size();
        tensor_index[tv] = index;
    }
    m_n_outputs = tensor_index.size() - m_n_inputs;

    // All remaining tensor views
    for (auto node : nodes)
    {
        for (auto output : node->get_outputs())
        {
            auto tv = output.get_tensor_view();
            if (0 == tensor_index.count(tv))
            {
                size_t index     = tensor_index.size();
                tensor_index[tv] = index;
                m_temp_views.push_back(tv);
            }
        }
    }

    // Now we build the eigen-VM instructions
    auto op_map = get_op_map();
    for (auto node : nodes)
    {
        auto handler_it = op_map.find(type_index(typeid(*node)));
        if (handler_it == op_map.end())
        {
            throw ngraph_error("Unhandled op during code generation");
        }
        std::vector<size_t> in;
        for (auto input : node->get_inputs())
        {
            auto output = input.get_output();
            auto tv     = output.get_tensor_view();
            in.push_back(tensor_index.at(tv));
        }
        std::vector<size_t> out;
        for (auto output : node->get_outputs())
        {
            auto tv = output.get_tensor_view();
            out.push_back(tensor_index.at(tv));
        }
        handler_it->second(node, this, in, out);
    }
    m_instructions->push_back(make_shared<runtime::eigen::ReturnInstruction>());
}

shared_ptr<ngraph::runtime::CallFrame> ExternalFunction::make_call_frame()
{
    std::vector<std::shared_ptr<ngraph::runtime::PrimaryTensorView>> temps;
    for (auto tv : m_temp_views)
    {
        temps.push_back(ngraph::runtime::eigen::make_tensor_view(tv));
    }
    return make_shared<ngraph::runtime::CallFrame>(
        m_n_inputs, m_n_outputs, temps, 0, m_instructions);
}
