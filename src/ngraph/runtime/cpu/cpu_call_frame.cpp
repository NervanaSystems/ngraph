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

#include <algorithm>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/cpu_tracing.hpp"

using namespace std;
using namespace ngraph;

runtime::cpu::CPU_CallFrame::CPU_CallFrame(std::shared_ptr<CPU_ExternalFunction> external_function,
                                           EntryPoint compiled_function)
    : m_external_function(external_function)
    , m_compiled_function(compiled_function)
{
    setup_runtime_context();
}

runtime::cpu::CPU_CallFrame::~CPU_CallFrame()
{
    cleanup_runtime_context();
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
        m_compiled_function(inputs.data(), outputs.data(), ctx);
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
            "Error propagating layouts - tensor view and layout descriptor counts do not match");
    }
    for (size_t i = 0; i < tvs.size(); i++)
    {
        if (layouts[i] == nullptr)
        {
            throw ngraph_error(
                "Error propagating layouts - layout information missing from tensor view");
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

    if (std::getenv("NGRAPH_CPU_USE_TBB") != nullptr)
    {
        ctx->G = new tbb::flow::graph;
        const auto envParallelism = std::getenv("NGRAPH_INTER_OP_PARALLELISM");
        const auto parallelism = envParallelism == nullptr ? 1 : std::atoi(envParallelism);
        ctx->c = new tbb::global_control(tbb::global_control::max_allowed_parallelism, parallelism);
    }
}

void runtime::cpu::CPU_CallFrame::cleanup_runtime_context()
{
    delete[] ctx->op_durations;
    delete[] ctx->p_en;
    for (auto buffer : ctx->memory_buffers)
    {
        delete buffer;
    }
    if (std::getenv("NGRAPH_CPU_USE_TBB") != nullptr)
    {
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
    delete ctx;
}
