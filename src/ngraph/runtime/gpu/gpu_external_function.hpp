/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

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
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_ExternalFunction;
            class GPU_Emitter;
            class GPU_CallFrame;

            using OpFunction =
                std::function<void(GPU_ExternalFunction* external_function,
                                   codegen::CodeWriter&,
                                   const ngraph::Node*,
                                   const std::vector<GPU_TensorViewWrapper>& inputs,
                                   const std::vector<GPU_TensorViewWrapper>& outputs)>;

            using OpMap = std::unordered_map<std::type_index, OpFunction>;

            class GPU_ExternalFunction : public ngraph::runtime::ExternalFunction,
                                         public std::enable_shared_from_this<GPU_ExternalFunction>
            {
                friend class GPU_CallFrame;

            public:
                GPU_ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                     bool release_function = true);
                std::shared_ptr<ngraph::runtime::CallFrame> make_call_frame();

            protected:
                void compile();

                EntryPoint m_compiled_function;

            private:
                void emit_debug_function_entry(codegen::CodeWriter& writer,
                                               Node* node,
                                               const std::vector<GPU_TensorViewWrapper>& in,
                                               const std::vector<GPU_TensorViewWrapper>& out);
                void emit_debug_function_exit(codegen::CodeWriter& writer,
                                              Node* node,
                                              const std::vector<GPU_TensorViewWrapper>& in,
                                              const std::vector<GPU_TensorViewWrapper>& out);
                void handle_output_alias(
                    codegen::CodeWriter& writer,
                    const Node&,
                    const std::unordered_map<descriptor::TensorView*, std::vector<size_t>>&);

                std::unique_ptr<codegen::Compiler> m_compiler;
                std::unique_ptr<codegen::ExecutionEngine> m_execution_engine;
                bool m_emit_timing;
                std::unordered_map<std::string, std::string> m_variable_name_map;
            };
        }
    }
}
