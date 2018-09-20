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
#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view_wrapper.hpp"

#define EMIT_ARGS                                                                                  \
    runtime::gpu::GPU_ExternalFunction *external_function, codegen::CodeWriter &writer,            \
        const Node *node, const std::vector<runtime::gpu::GPU_TensorViewWrapper> &args,            \
        const std::vector<runtime::gpu::GPU_TensorViewWrapper> &out

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter;
            class GPU_CallFrame;
            struct GPURuntimeContext;

            class GPU_ExternalFunction : public std::enable_shared_from_this<GPU_ExternalFunction>
            {
                friend class GPU_CallFrame;
                friend class GPU_Backend;

            public:
                GPU_ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                     std::shared_ptr<GPU_Backend::BackendContext>& shared_context,
                                     bool release_function = true);
                ~GPU_ExternalFunction();

                std::shared_ptr<ngraph::runtime::gpu::GPU_CallFrame> make_call_frame();
                std::unique_ptr<runtime::gpu::GPURuntimeContext>& ctx();
                const std::unique_ptr<GPUPrimitiveEmitter>& get_primitive_emitter() const
                {
                    return m_shared_context->m_primitive_emitter;
                }

                static const size_t s_memory_pool_alignment;

            protected:
                void compile();

                EntryPoint m_compiled_function;

            private:
                // For non-destructive passthrough kernels, propagate function
                // input buffers to internal ops
                void propagate_in_place_input(ngraph::descriptor::Output* output,
                                              std::string input_name);
                // For in-place kernels, propagate function output buffers to
                // internal ops
                void propagate_in_place_output(ngraph::descriptor::Output* res_src_output,
                                               std::string output_name);
                void emit_header();
                void emit_timer_functions();
                void emit_constant_declarations();
                void emit_function_declarations();
                void emit_functions();
                void emit_debug_function_entry(Node* node);
                void emit_debug_function_exit(Node* node);
                void emit_temp_mem_pool_allocation(std::shared_ptr<Function> current_function);
                void emit_op(EMIT_ARGS);
                void release_function() { m_function = nullptr; }
                void store_emitted_functions(const std::string& code);
                std::string emit_op_as_function(const Node& node, const std::string& function_name);
                std::string strip_comments(const std::string& s) const;

                codegen::CodeWriter m_writer;
                ngraph::pass::Manager m_pass_manager;

                std::unique_ptr<codegen::Compiler> m_compiler;
                std::unique_ptr<codegen::ExecutionEngine> m_execution_engine;
                std::shared_ptr<ngraph::Function> m_function;

                std::map<std::string, size_t> m_name_index_map;
                std::unordered_map<std::string, std::string> m_variable_name_map;
                std::unordered_map<Node*, Node*> m_node_function_map;
                std::unordered_map<std::shared_ptr<Function>, std::list<std::shared_ptr<Node>>>
                    m_function_ordered_ops;

                bool m_emit_timing;
                bool m_is_compiled;
                bool m_release_function;
                bool m_temporaries_used;
                size_t m_offset;

                std::string m_function_name;
                std::string m_pch_header_source;

                std::shared_ptr<std::unordered_map<std::string, size_t>> m_tensor_memory_buffers;
                std::shared_ptr<GPU_Backend::BackendContext> m_shared_context;
            };
        }
    }
}
