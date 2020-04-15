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

#pragma once

#if !defined(NGRAPH_DEX_ONLY)

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#include "ngraph/function.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_compiled_function.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter;
            struct GPURuntimeContext;

            class GPUExternalFunction : public GPUCompiledFunction
            {
            public:
                GPUExternalFunction(
                    const std::shared_ptr<ngraph::Function>& function,
                    const std::shared_ptr<GPUBackend::BackendContext>& shared_context);
                virtual ~GPUExternalFunction();

                virtual std::string
                    add_to_runtime(size_t primitive_index,
                                   const std::string& function_name,
                                   const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                                   const std::vector<runtime::gpu::GPUTensorWrapper>& out) override;
                virtual std::string add_call_to_runtime(
                    const std::string& caller,
                    const std::string& callee,
                    const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                    const std::vector<runtime::gpu::GPUTensorWrapper>& out) override;
                virtual void get_performance_data(
                    std::vector<runtime::PerformanceCounter>& rc) const override;

            protected:
                virtual void compile_function() override;
                virtual void add_passes(ngraph::pass::Manager& pass_manager) override;
                virtual void emit() override;

            private:
                /// \brief Create a list of node names for each arg in args
                /// \param args list of tensor arguments
                /// \param arg_indexes a list of indexes into args for which args to include in
                ///    the output list, so {1, 2} will include args 1 and 2 and skip 0.
                /// \ return returns a string containing "arg0_name, arg1_name, etc."
                std::string node_names(const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                                       std::initializer_list<int> arg_indexes = {});

                void emit_header();
                void emit_timer_functions();
                void emit_constant_declarations();
                void emit_function_declarations();
                void emit_functions();
                void emit_debug_function_entry(Node* node);
                void emit_debug_function_exit(Node* node);
                void emit_temp_mem_pool_allocation(std::shared_ptr<Function> current_function);
                void store_emitted_functions(const std::string& code);
                std::string emit_op(EMIT_ARGS);
                std::string emit_op_as_function(const Node& node, const std::string& function_name);
                std::string strip_comments(const std::string& s) const;

                static const std::string& get_pch_header_source();
                static const std::string& get_header_source();

                // For non-destructive passthrough kernels, propagate function
                // input buffers to internal ops
                virtual void propagate_in_place_input(ngraph::descriptor::Output* output,
                                                      const std::string& input_name) override;
                // For in-place kernels, propagate function output buffers to
                // internal ops
                virtual void propagate_in_place_output(ngraph::descriptor::Output* res_src_output,
                                                       const std::string& output_name) override;
                CodeWriter m_writer;
                std::string m_common_function_string;
                std::unique_ptr<codegen::Compiler> m_compiler;
                std::unique_ptr<codegen::ExecutionEngine> m_execution_engine;
                std::map<std::string, size_t> m_name_index_map;
                std::unordered_map<std::string, std::string> m_variable_name_map;
                std::unordered_map<Node*, Node*> m_node_function_map;
            };
        }
    }
}
#endif // !defined(NGRAPH_DEX_ONLY)
