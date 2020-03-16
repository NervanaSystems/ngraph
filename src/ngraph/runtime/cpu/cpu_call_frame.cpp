//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include <thread>

#include "ngraph/env_util.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor.hpp"
#include "ngraph/runtime/cpu/cpu_tracing.hpp"
#include "ngraph/runtime/cpu/mkldnn_emitter.hpp"

using namespace std;
using namespace ngraph;

runtime::cpu::CPU_CallFrame::CPU_CallFrame(std::shared_ptr<CPU_ExternalFunction> external_function,
                                           InitContextFuncCG compiled_init_ctx_func,
                                           DestroyContextFuncCG compiled_destroy_ctx_func,
                                           EntryPoint compiled_function,
                                           runtime::Allocator* allocator)
    : m_external_function(external_function)
    , m_compiled_init_ctx_func(compiled_init_ctx_func)
    , m_compiled_destroy_ctx_func(compiled_destroy_ctx_func)
    , m_compiled_function(compiled_function)
{
    const auto envConcurrency = getenv_int("NGRAPH_CPU_CONCURRENCY");
    m_num_ctx = envConcurrency <= 0 ? 1 : envConcurrency;
    if (m_num_ctx > std::thread::hardware_concurrency())
    {
        throw ngraph_error(
            "Unexpected value specified for NGRAPH_CPU_CONCURRENCY "
            "(" +
            std::to_string(envConcurrency) + "). Please specify a value in range [1-" +
            std::to_string(std::thread::hardware_concurrency()) + "]");
    }

    setup_runtime_context(allocator);
    if (!m_external_function->is_direct_execution())
    {
        // Invoke codegen runtime context initialization function.
        NGRAPH_CHECK(m_compiled_init_ctx_func, "compiled_init_ctx_func cannot be null.");
        cg_ctx = m_compiled_init_ctx_func();
    }
}

runtime::cpu::CPU_CallFrame::~CPU_CallFrame()
{
    cleanup_runtime_context();
    if (!m_external_function->is_direct_execution())
    {
        m_compiled_destroy_ctx_func(cg_ctx);
    }
}

void runtime::cpu::CPU_CallFrame::inner_call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& output_tvs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& input_tvs,
    const size_t id,
    const bool disable_caching)
{
    vector<void*> inputs;
    vector<void*> outputs;

    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::cpu::CPUTensor> tv =
            static_pointer_cast<runtime::cpu::CPUTensor>(input_tvs[i]);
        if (disable_caching)
        {
            m_ctx_vec[id]->p_en[i] = true;
        }
        else
        {
            m_ctx_vec[id]->p_en[i] = tv->get_stale();
        }

        inputs.push_back(tv->get_data_ptr());
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::cpu::CPUTensor> tv =
            static_pointer_cast<runtime::cpu::CPUTensor>(output_tvs[i]);
        outputs.push_back(tv->get_data_ptr());
    }

    // Invoke compiled computation
    if (!m_external_function->is_direct_execution())
    {
        m_compiled_function(inputs.data(), outputs.data(), m_ctx_vec[id], cg_ctx);
    }
    else
    {
        m_external_function->get_executor()(m_ctx_vec[id], inputs, outputs);
    }

    if (runtime::cpu::IsTracingEnabled())
    {
        GenerateTimeline(m_external_function->get_op_attrs(),
                         m_ctx_vec[id]->op_durations,
                         m_external_function->get_function_name() + ".timeline.json");
    }
}

void runtime::cpu::CPU_CallFrame::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& output_tvs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& input_tvs)
{
    size_t id = 0;
    auto disable_caching = false;
    {
        std::unique_lock<std::mutex> lck(m_mutex);
        while (m_num_ctx_available == 0)
        {
            m_cv.wait(lck);
        }

        for (size_t i = 0; i < m_num_ctx; i++)
        {
            if (m_id_pool[i])
            {
                id = i;
                break;
            }
        }
        NGRAPH_CHECK(id != m_num_ctx);
        m_id_pool[id] = false;
        if (id != m_prev_ctx)
        {
            // Disable caching since staleness hints are no longer
            // applicable to this context
            disable_caching = true;
        }
        m_prev_ctx = id;
        m_num_ctx_available--;
    }

    m_ctx_vec[id]->pc = 0;
    propagate_layouts(output_tvs, m_external_function->get_result_layout_descriptors());
    inner_call(output_tvs, input_tvs, id, disable_caching);

    m_mutex.lock();
    m_id_pool[id] = true;
    m_num_ctx_available++;
    m_mutex.unlock();
    m_cv.notify_one();
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

void runtime::cpu::CPU_CallFrame::setup_runtime_context(Allocator* allocator)
{
    for (size_t i = 0; i < m_num_ctx; i++)
    {
        m_id_pool[i] = true;
        auto ctx = new CPURuntimeContext;
        m_ctx_vec.push_back(ctx);

        ctx->pc = 0;
        ctx->op_durations = nullptr;
        if (runtime::cpu::IsTracingEnabled())
        {
            ctx->op_durations = new int64_t[m_external_function->get_op_attrs().size()];
        }
        ctx->p_en = new bool[m_external_function->get_parameter_layout_descriptors().size()];

        ctx->first_iteration = true;

        ctx->buffer_data = std::vector<void*>(m_external_function->get_buffer_size());

        // Create temporary buffer pools
        size_t alignment = runtime::cpu::CPU_ExternalFunction::s_memory_pool_alignment;
        for (auto buffer_size : m_external_function->get_memory_buffer_sizes())
        {
            auto buffer = new AlignedBuffer(buffer_size, alignment, allocator);
            ctx->memory_buffers.push_back(buffer);
        }
        const auto& mkldnn_emitter = m_external_function->get_mkldnn_emitter();
        // Create scratchpad
        auto scratchpad_size = mkldnn_emitter->get_max_scratchpad_size();
        if (m_external_function->is_direct_execution())
        {
            ctx->mkldnn_primitives =
                std::vector<mkldnn::primitive*>(mkldnn_emitter->get_mkldnn_primitives().size());
            ctx->mkldnn_memories =
                std::vector<mkldnn::memory*>(mkldnn_emitter->get_mkldnn_memories().size());
            ctx->mkldnn_scratchpad_mds = std::vector<mkldnn::memory::desc*>(
                mkldnn_emitter->get_mkldnn_scratchpad_mds().size());
            if (scratchpad_size > 0)
            {
                ctx->scratchpad_buffer = new AlignedBuffer(scratchpad_size, alignment, allocator);
            }
            else
            {
                ctx->scratchpad_buffer = nullptr;
            }
        }
        else
        {
            // single thread for codegen
            NGRAPH_CHECK(m_num_ctx == 1);
        }

        ctx->states = m_external_function->m_states.data();
#if defined(NGRAPH_TBB_ENABLE)
        if (m_external_function->is_direct_execution() && getenv_bool("NGRAPH_CPU_USE_TBB"))
        {
            // For codegen mode, graph and global control are now part of the code generated
            // CPURuntimeContextCG class.
            ctx->G = new tbb::flow::graph;
            const auto envParallelism = getenv_int("NGRAPH_INTER_OP_PARALLELISM");
            const auto parallelism = envParallelism <= 0 ? 1 : envParallelism;
            ctx->c =
                new tbb::global_control(tbb::global_control::max_allowed_parallelism, parallelism);
        }
#endif
    }
    m_num_ctx_available = m_num_ctx;
}

void runtime::cpu::CPU_CallFrame::cleanup_runtime_context()
{
    for (size_t i = 0; i < m_num_ctx; i++)
    {
        auto ctx = m_ctx_vec.back();
        m_ctx_vec.pop_back();

        delete[] ctx->op_durations;
        delete[] ctx->p_en;
        for (auto p : ctx->mkldnn_primitives)
        {
            delete p;
        }
        for (auto m : ctx->mkldnn_memories)
        {
            delete m;
        }
        for (auto buffer : ctx->memory_buffers)
        {
            delete buffer;
        }
        for (auto s : ctx->mkldnn_scratchpad_mds)
        {
            delete s;
        }
        if (m_external_function->is_direct_execution())
        {
            delete ctx->scratchpad_buffer;
        }

#if defined(NGRAPH_TBB_ENABLE)
        if (m_external_function->is_direct_execution() && getenv_bool("NGRAPH_CPU_USE_TBB"))
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
#endif
        delete ctx;
    }
    m_num_ctx_available = 0;
}
