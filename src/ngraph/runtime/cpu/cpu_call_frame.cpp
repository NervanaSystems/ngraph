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

#include <algorithm>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/cpu_tracing.hpp"

#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
#include <mlsl.hpp>
#endif

using namespace std;
using namespace ngraph;

runtime::cpu::CPU_CallFrame::CPU_CallFrame(std::shared_ptr<CPU_ExternalFunction> external_function,
                                           InitContextFuncCG compiled_init_ctx_func,
                                           DestroyContextFuncCG compiled_destroy_ctx_func,
                                           EntryPoint compiled_function)
    : m_external_function(external_function)
    , m_compiled_init_ctx_func(compiled_init_ctx_func)
    , m_compiled_destroy_ctx_func(compiled_destroy_ctx_func)
    , m_compiled_function(compiled_function)
{
    setup_runtime_context();
    if (!m_external_function->is_direct_execution())
    {
        // Invoke codegen runtime context initialization function.
        NGRAPH_ASSERT(m_compiled_init_ctx_func) << "compiled_init_ctx_func cannot be null.";
        cg_ctx = m_compiled_init_ctx_func();
    }
}

runtime::cpu::CPU_CallFrame::~CPU_CallFrame()
{
    if (!m_external_function->is_direct_execution())
    {
        NGRAPH_ASSERT(m_compiled_destroy_ctx_func) << "compiled_destroy_ctx_func cannot be null.";
        m_compiled_destroy_ctx_func(cg_ctx);
    }
}

void runtime::cpu::CPU_CallFrame::inner_call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& output_tvs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& input_tvs)
{
    vector<void*> inputs;
    vector<void*> outputs;

    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::cpu::CPUTensorView> tv =
            static_pointer_cast<runtime::cpu::CPUTensorView>(input_tvs[i]);
        ctx->p_en[i] = tv->get_stale();
        inputs.push_back(tv->get_data_ptr());
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::cpu::CPUTensorView> tv =
            static_pointer_cast<runtime::cpu::CPUTensorView>(output_tvs[i]);
        outputs.push_back(tv->get_data_ptr());
    }

    // Invoke compiled computation
    if (!m_external_function->is_direct_execution())
    {
        m_compiled_function(inputs.data(), outputs.data(), ctx, cg_ctx);
    }
    else
    {
        m_external_function->get_executor()(ctx, inputs, outputs);
    }

    if (runtime::cpu::IsTracingEnabled())
    {
        GenerateTimeline(m_external_function->get_op_attrs(),
                         ctx->op_durations,
                         m_external_function->get_function_name() + ".timeline.json");
    }
}

void runtime::cpu::CPU_CallFrame::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& output_tvs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& input_tvs)
{
    ctx->pc = 0;
    propagate_layouts(output_tvs, m_external_function->get_result_layout_descriptors());
    inner_call(output_tvs, input_tvs);
}

void runtime::cpu::CPU_CallFrame::propagate_layouts(
    const std::vector<std::shared_ptr<runtime::Tensor>>& tvs,
    const LayoutDescriptorPtrs& layouts) const
{
    if (layouts.size() != tvs.size())
    {
        throw ngraph_error(
            "Error propagating layouts - tensor and layout descriptor counts do not match");
    }
    for (size_t i = 0; i < tvs.size(); i++)
    {
        if (layouts[i] == nullptr)
        {
            throw ngraph_error(
                "Error propagating layouts - layout information missing from tensor");
        }
        tvs[i]->set_tensor_layout(layouts[i]);
    }
}

void runtime::cpu::CPU_CallFrame::setup_runtime_context()
{
    ctx = new CPURuntimeContext;

    ctx->pc = 0;
    ctx->op_durations = nullptr;
    if (runtime::cpu::IsTracingEnabled())
    {
        ctx->op_durations = new int64_t[m_external_function->get_op_attrs().size()];
    }
    ctx->p_en = new bool[m_external_function->get_parameter_layout_descriptors().size()];

    ctx->first_iteration = true;

    // Create temporary buffer pools
    size_t alignment = runtime::cpu::CPU_ExternalFunction::s_memory_pool_alignment;
    for (auto buffer_size : m_external_function->get_memory_buffer_sizes())
    {
        auto buffer = new AlignedBuffer(buffer_size, alignment);
        ctx->memory_buffers.push_back(buffer);
    }
    const auto& mkldnn_emitter = m_external_function->get_mkldnn_emitter();
    ctx->mkldnn_primitives = mkldnn_emitter->get_mkldnn_primitives().data();
    ctx->mkldnn_workspaces = mkldnn_emitter->get_mkldnn_workspaces().data();
    ctx->states = m_external_function->m_states.data();

    if (m_external_function->is_direct_execution() && std::getenv("NGRAPH_CPU_USE_TBB") != nullptr)
    {
        // For codegen mode, graph and global control are now part of the code generated
        // CPURuntimeContextCG class.
        ctx->G = new tbb::flow::graph;
        const auto envParallelism = std::getenv("NGRAPH_INTER_OP_PARALLELISM");
        const auto parallelism = envParallelism == nullptr ? 1 : std::atoi(envParallelism);
        ctx->c = new tbb::global_control(tbb::global_control::max_allowed_parallelism, parallelism);
    }

#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
    if (MLSL::Environment::GetEnv().IsInitialized())
    {
        ctx->mlsl_env = &MLSL::Environment::GetEnv();
        ctx->mlsl_dist = ctx->mlsl_env->CreateDistribution(ctx->mlsl_env->GetProcessCount(), 1);
    }
#endif
}

void runtime::cpu::CPU_CallFrame::cleanup_runtime_context()
{
    delete[] ctx->op_durations;
    delete[] ctx->p_en;
    for (auto buffer : ctx->memory_buffers)
    {
        delete buffer;
    }
    if (m_external_function->is_direct_execution() && std::getenv("NGRAPH_CPU_USE_TBB") != nullptr)
    {
        // For codegen mode, graph and global control are now part of a code generated
        // CPURuntimeContext class.

        // delete graph G and nodes in G
        ctx->G->wait_for_all();
        std::vector<tbb::flow::graph_node*> to_be_deleted;
        for (auto it = ctx->G->begin(); it != ctx->G->end(); it++)
        {
            to_be_deleted.push_back(&(*it));
        }
        delete ctx->G;
        for (auto node : to_be_deleted)
        {
            delete node;
        }
        delete ctx->c;
    }

#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
    if (MLSL::Environment::GetEnv().IsInitialized() && ctx->mlsl_dist != nullptr)
    {
        ctx->mlsl_env->DeleteDistribution(ctx->mlsl_dist);
    }
#endif
    delete ctx;
}
