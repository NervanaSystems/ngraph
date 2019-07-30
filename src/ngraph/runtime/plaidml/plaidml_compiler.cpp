//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pass/zero_dim_tensor_elimination.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"
#include "ngraph/runtime/plaidml/plaidml_logger.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_concat_elision.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_concat_split.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_explicit_logicals.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_implicit_broadcast.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_lower_convolutions.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_replicate_combination.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_replicate_elision.hpp"
#include "ngraph/runtime/plaidml/plaidml_pass_winograd.hpp"

namespace
{
    void write_debug(const ngraph::Node& op)
    {
        PLAIDML_DEBUG << "Compiling: " << op;
        for (const auto& op_input : op.get_inputs())
        {
            ngraph::descriptor::Tensor* tensor = op_input.get_output().get_tensor_ptr().get();
            PLAIDML_DEBUG << "Input: descriptor::Tensor " << tensor << " "
                          << op.get_input_shape(op_input.get_index())
                          << op.get_input_element_type(op_input.get_index());
        }
        for (std::size_t out_idx = 0; out_idx < op.get_output_size(); ++out_idx)
        {
            ngraph::descriptor::Tensor* tensor = op.get_output_tensor_ptr(out_idx).get();
            PLAIDML_DEBUG << "Output: descriptor::Tensor " << tensor << " "
                          << op.get_output_shape(out_idx) << op.get_output_element_type(out_idx);
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
}

std::shared_ptr<ngraph::runtime::plaidml::PlaidML_Executable>
    ngraph::runtime::plaidml::Compiler::compile(std::shared_ptr<Function> func)
{
    // N.B. ngraph::pass::Manager::run_passes() is *not* a const
    // method; it mutates the manager, possibly causing corruption if
    // multiple threads are simultaneously running passes.  So we
    // build a new manager for each compilation, instead of building
    // one when we build the Compiler and reusing it for each
    // compilation.

    ngraph::pass::Manager pass_manager;

    // We apply the same general-purposes passes as the CPU backend.
    pass_manager.register_pass<ngraph::pass::FusedOpDecomposition>();
    pass_manager.register_pass<ngraph::pass::LikeReplacement>();
    pass_manager.register_pass<ngraph::pass::NopElimination>();
    pass_manager.register_pass<ngraph::pass::ZeroDimTensorElimination>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.register_pass<ngraph::pass::CoreFusion>();
    // N.B. We'd like to register ngraph::pass::GetOutputElementElimination, but it breaks BatchNorm
    // backprop
    pass_manager.register_pass<ngraph::pass::Liveness>();
    pass_manager.register_pass<ngraph::runtime::plaidml::pass::ExplicitLogicals>();
    pass_manager.register_pass<ngraph::runtime::plaidml::pass::ConcatElision>();
    pass_manager.register_pass<ngraph::runtime::plaidml::pass::ConcatSplit>();
    pass_manager.register_pass<ngraph::runtime::plaidml::pass::ReplicateElision>();
    pass_manager.register_pass<ngraph::runtime::plaidml::pass::ReplicateCombination>();
    pass_manager.register_pass<ngraph::runtime::plaidml::pass::ImplicitBroadcast>();
    pass_manager.register_pass<ngraph::runtime::plaidml::pass::LowerConvolutions>();
    if (pass_manager.get_pass_config().get_pass_enable("Winograd"))
    {
        pass_manager.register_pass<ngraph::runtime::plaidml::pass::Winograd>();
    }
    if (!m_config->graphviz.empty())
    {
        pass_manager.register_pass<ngraph::pass::VisualizeTree>(m_config->graphviz);
    }

    // N.B. When we rewrite the graph, there are cases where we
    // produce nodes that contain validation errors.  A good example
    // is in the ImplicitBroadcast pass -- after this pass, there may
    // be elementwise operations whose inputs are not all the same
    // shape.
    //
    // The caller may wish to perform operations (e.g. clone) on their
    // supplied function that will cause validation to occur.  So
    // before we rewrite, we make our own copy of the function.
    auto rewrite_func = clone_function(*func);

    // Apply passes, with revalidation disabled.
    pass_manager.run_passes(rewrite_func, true, false);

    // Compile the resulting function.
    Build b;
    build(std::move(rewrite_func), &b);
    return std::make_shared<PlaidML_Executable>(std::move(b), std::move(func));
}

bool ngraph::runtime::plaidml::Compiler::is_supported(const Node& node) const
{
    return GlobalOpImplMap()->count(std::type_index(typeid(node))) != 0;
}

void ngraph::runtime::plaidml::Compiler::build(std::shared_ptr<Function> func, Build* b)
{
    b->compiler = this;
    b->config = m_config;
    b->func = func;

    const auto* op_map = GlobalOpImplMap();

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
        it->second->Apply(b, op);
    }
}
