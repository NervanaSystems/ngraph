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

#pragma once

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#include "ngraph/function.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"

#define EMIT_ARGS                                                                                  \
    runtime::gpu::GPU_CompiledFunction *compiled_function,            \
        const Node *node, const std::vector<runtime::gpu::GPUTensorWrapper> &args,                 \
        const std::vector<runtime::gpu::GPUTensorWrapper> &out

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter;
            struct GPURuntimeContext;

            class GPU_CompiledFunction
            {
                friend class GPU_Backend;

            public:
                GPU_CompiledFunction(const std::shared_ptr<ngraph::Function>& function,
                                         std::shared_ptr<GPU_Backend::BackendContext>& shared_context);
                virtual ~GPU_CompiledFunction();

                std::unique_ptr<runtime::gpu::GPURuntimeContext>& ctx();
                const std::unique_ptr<GPUPrimitiveEmitter>& get_primitive_emitter() const
                {
                    return m_shared_context->m_primitive_emitter;
                }
                virtual std::string add_to_runtime(size_t primitive_index,
                                                   const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                                                   const std::vector<runtime::gpu::GPUTensorWrapper>& out) = 0;
                virtual void compile() = 0;
                virtual void get_performance_data(std::vector<runtime::PerformanceCounter>& rc) const = 0;

                static const size_t s_memory_pool_alignment;
                static const std::string s_output_dir;
            protected:
                EntryPoint m_runtime;

                // For non-destructive passthrough kernels, propagate function
                // input buffers to internal ops
                virtual void propagate_in_place_input(ngraph::descriptor::Output* output,
                                                      std::string input_name) = 0;
                // For in-place kernels, propagate function output buffers to
                // internal ops
                virtual void propagate_in_place_output(ngraph::descriptor::Output* res_src_output,
                                                       std::string output_name) = 0;
                std::shared_ptr<ngraph::Function> m_function;

                std::unordered_map<std::shared_ptr<Function>, std::list<std::shared_ptr<Node>>>
                    m_function_ordered_ops;

                bool m_emit_timing;
                bool m_is_compiled;
                size_t m_offset;

                std::string m_function_name;

                std::unordered_map<std::string, size_t> m_tensor_memory_buffers;
                std::shared_ptr<GPU_Backend::BackendContext> m_shared_context;
            };
        }
    }
}
