//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/runtime/plaidml/plaidml_compiler.hpp"
#include "ngraph/log.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"
#include "ngraph/runtime/plaidml/plaidml_logger.hpp"

namespace
{
    void write_debug(const ngraph::Node& op)
    {
        PLAIDML_DEBUG << "Node: name=\"" << op.get_name() << "\" desc=\"" << op.description()
                      << "\"";
        for (const auto& op_input : op.get_inputs())
        {
            ngraph::descriptor::Tensor* tensor = op_input.get_output().get_tensor_ptr().get();
            PLAIDML_DEBUG << "Input: descriptor::Tensor " << tensor << " "
                          << op.get_input_shape(op_input.get_index());
        }
        for (std::size_t out_idx = 0; out_idx < op.get_output_size(); ++out_idx)
        {
            ngraph::descriptor::Tensor* tensor = op.get_output_tensor_ptr(out_idx).get();
            PLAIDML_DEBUG << "Output: descriptor::Tensor " << tensor << " "
                          << op.get_output_shape(out_idx);
        }
        for (auto* t : op.liveness_new_list)
        {
            PLAIDML_DEBUG << "New tensor: " << t;
        }
        for (auto* t : op.liveness_free_list)
        {
            PLAIDML_DEBUG << "Retire tensor: " << t;
        }
    }
}

ngraph::runtime::plaidml::Compiler::Compiler(Config* config)
    : m_config{config}
{
    // We apply the same general-purposes passes as the CPU backend.
    m_pass_manager.register_pass<ngraph::pass::LikeReplacement>();
    m_pass_manager.register_pass<ngraph::pass::NopElimination>();
    m_pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    m_pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    m_pass_manager.register_pass<ngraph::pass::CoreFusion>();
    // N.B. We'd like to register ngraph::pass::GetOutputElementElimination, but it breaks BatchNorm
    // backprop
    m_pass_manager.register_pass<ngraph::pass::Liveness>();
}

std::shared_ptr<ngraph::runtime::plaidml::CompiledFunction>
    ngraph::runtime::plaidml::Compiler::compile(std::shared_ptr<Function> func)
{
    m_pass_manager.run_passes(func);

    Build b;
    build(std::move(func), &b);
    return std::make_shared<CompiledFunction>(std::move(b));
}

void ngraph::runtime::plaidml::Compiler::build(std::shared_ptr<Function> func, Build* b)
{
    b->compiler = this;
    b->config = m_config;
    b->func = func;

    const auto* op_map = OpImplMap();

    for (const auto& op_ptr : func->get_ordered_ops())
    {
        const ngraph::Node* op = op_ptr.get();
        if (m_config->debug)
        {
            write_debug(*op);
        }
        auto it = op_map->find(std::type_index(typeid(*op)));
        if (it == op_map->end())
        {
            throw unsupported_op{
                std::string{"The PlaidML backend doesn't currently implement the '"} +
                op->description() + "' operation"};
        }
        it->second(b, *op);
    }
}
