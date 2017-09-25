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
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/eigen/abs.hpp"
#include "ngraph/runtime/eigen/add.hpp"
#include "ngraph/runtime/eigen/constant.hpp"
#include "ngraph/runtime/eigen/divide.hpp"
#include "ngraph/runtime/eigen/equal.hpp"
#include "ngraph/runtime/eigen/less_than.hpp"
#include "ngraph/runtime/eigen/log.hpp"
#include "ngraph/runtime/eigen/maximum.hpp"
#include "ngraph/runtime/eigen/multiply.hpp"
#include "ngraph/runtime/eigen/negate.hpp"
#include "ngraph/runtime/eigen/not_equal.hpp"
#include "ngraph/runtime/eigen/return.hpp"
#include "ngraph/runtime/eigen/select.hpp"
#include "ngraph/runtime/eigen/subtract.hpp"
#include "ngraph/runtime/utils.hpp"

using namespace std;
using namespace ngraph::runtime;

ExternalFunction::ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                   bool                                     release_function)
    : m_function(function)
    , m_release_function(release_function)
    , m_is_compiled(false)
    , m_instructions(make_shared<std::vector<std::shared_ptr<ngraph::runtime::Instruction>>>())
{
}

#define REGISTER_INSTRUCTION(op_class,instr_class,...)                          \
    op_map[type_index(typeid(op_class))] = [](Node *                      n,    \
                                              ExternalFunction*          ef,    \
                                              const std::vector<size_t>& in,    \
                                              const std::vector<size_t>& out) { \
            ef->get_instructions()->push_back(                                  \
                make_shared<instr_class>(__VA_ARGS__));                         \
    }

#define REGISTER_UNOP(op_class,instr_class) \
    REGISTER_INSTRUCTION(op_class,instr_class,in[0],out[0])
#define REGISTER_BINOP(op_class,instr_class) \
    REGISTER_INSTRUCTION(op_class,instr_class,in[0],in[1],out[0])
#define REGISTER_TERNOP(op_class,instr_class) \
    REGISTER_INSTRUCTION(op_class,instr_class,in[0],in[1],in[2],out[0])

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
        REGISTER_UNOP  (op::Abs,     runtime::eigen::AbsInstruction<element::Float32>);
        REGISTER_BINOP (op::Add,     runtime::eigen::AddInstruction<element::Float32>);
        REGISTER_BINOP (op::Divide,  runtime::eigen::DivideInstruction<element::Float32>);
        REGISTER_BINOP (op::Equal,   runtime::eigen::EqualInstruction<element::Float32>);
        REGISTER_BINOP (op::Less,    runtime::eigen::LessThanInstruction<element::Float32>);
        REGISTER_UNOP  (op::Log,     runtime::eigen::LogInstruction<element::Float32>);
        REGISTER_BINOP (op::Maximum, runtime::eigen::MaximumInstruction<element::Float32>);
        REGISTER_BINOP (op::Multiply,runtime::eigen::MultiplyInstruction<element::Float32>);
        REGISTER_UNOP  (op::Negative,runtime::eigen::NegateInstruction<element::Float32>);
        REGISTER_BINOP (op::NotEqual,runtime::eigen::NotEqualInstruction<element::Float32>);
        REGISTER_TERNOP(op::Select,  runtime::eigen::SelectInstruction<element::Float32>);
        REGISTER_BINOP (op::Subtract,runtime::eigen::SubtractInstruction<element::Float32>);

        // Parameter, as a "runtime no-op", is a special case.
        op_map[type_index(typeid(op::Parameter))] = [](Node*                      n,
                                                       ExternalFunction*          ef,
                                                       const std::vector<size_t>& in,
                                                       const std::vector<size_t>& out) {};

        REGISTER_INSTRUCTION(op::ScalarConstant<element::Float32>,
                             runtime::eigen::ConstantInstruction<element::Float32>,
                             std::vector<element::Float32::type>{dynamic_cast<op::ScalarConstant<element::Float32>*>(n)->get_value()},
                             out[0]);

        REGISTER_INSTRUCTION(op::TensorConstant<element::Float32>,
                             runtime::eigen::ConstantInstruction<element::Float32>,
                             dynamic_cast<op::TensorConstant<element::Float32>*>(n)->get_value()->get_vector(),
                             out[0]);

        initialized = true;
    }
    return op_map;
}

void ExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }

    // This will be replaced with the pass manager
    // Get the ordered list of ops in execution order
    pass::Manager pass_manager;
    auto          topological_sort = make_shared<pass::TopologicalSort>();
    pass_manager.register_pass(topological_sort);
    pass_manager.run_passes(m_function);
    auto nodes = pass_manager.get_call_graph();
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
    for (auto param : m_function->get_parameters())
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
    for (auto output : m_function->get_result()->get_outputs())
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
    m_is_compiled = true;
    if (m_release_function)
    {
        release_function();
    }
}

shared_ptr<ngraph::runtime::CallFrame> ExternalFunction::make_call_frame()
{
    if (!m_is_compiled)
    {
        compile();
    }
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> temps;
    for (auto tv : m_temp_views)
    {
        temps.push_back(ngraph::runtime::make_tensor<ngraph::element::Float32>(tv->get_tensor_view_type()->get_shape()));
    }
    return make_shared<ngraph::runtime::CallFrame>(
        m_n_inputs, m_n_outputs, temps, 0, m_instructions);
}
