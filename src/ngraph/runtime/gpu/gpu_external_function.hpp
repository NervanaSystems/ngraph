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
#include "ngraph/runtime/gpu/gpu_external_function_base.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter;
            struct GPURuntimeContext;

            class GPU_ExternalFunction : public GPU_ExternalFunctionBase
            {
            public:
                GPU_ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                     std::shared_ptr<GPU_Backend::BackendContext>& shared_context);
                virtual ~GPU_ExternalFunction();

                virtual std::string add_to_runtime(size_t primitive_index,
                                            const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                                            const std::vector<runtime::gpu::GPUTensorWrapper>& out) override;
                virtual void compile() override;
                virtual void get_performance_data(std::vector<runtime::PerformanceCounter>& rc) const override;
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
                                                      std::string input_name) override;
                // For in-place kernels, propagate function output buffers to
                // internal ops
                virtual void propagate_in_place_output(ngraph::descriptor::Output* res_src_output,
                                                       std::string output_name) override;
                codegen::CodeWriter m_writer;
                std::unique_ptr<codegen::Compiler> m_compiler;
                std::unique_ptr<codegen::ExecutionEngine> m_execution_engine;
                std::map<std::string, size_t> m_name_index_map;
                std::unordered_map<std::string, std::string> m_variable_name_map;
                std::unordered_map<Node*, Node*> m_node_function_map;
            };
        }
    }
}
