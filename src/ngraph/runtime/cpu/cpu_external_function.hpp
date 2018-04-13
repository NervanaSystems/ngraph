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
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#include "ngraph/function.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_emitter.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class CPU_Emitter;
            class CPU_CallFrame;

            using OpFunction = std::function<void(CPU_ExternalFunction* external_function,
                                                  codegen::CodeWriter&,
                                                  const ngraph::Node*,
                                                  const std::vector<TensorViewWrapper>& inputs,
                                                  const std::vector<TensorViewWrapper>& outputs)>;

            using OpMap = std::unordered_map<std::type_index, OpFunction>;

            struct OpAttributes
            {
                std::string Description;
                std::vector<std::string> Outputs;
                std::vector<std::string> Inputs;
                OpAttributes(const std::string& desc,
                             const std::vector<std::string>& outputs,
                             const std::vector<std::string>& inputs)
                    : Description(desc)
                    , Outputs(outputs)
                    , Inputs(inputs)
                {
                }
            };

            class CPU_ExternalFunction : public std::enable_shared_from_this<CPU_ExternalFunction>
            {
                friend class CPU_Backend;

            public:
                CPU_ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                     bool release_function = true);
                ~CPU_ExternalFunction();
                std::shared_ptr<ngraph::runtime::cpu::CPU_CallFrame> make_call_frame();

                const LayoutDescriptorPtrs& get_parameter_layout_descriptors();
                const LayoutDescriptorPtrs& get_result_layout_descriptors();

                const std::vector<OpAttributes>& get_op_attrs() const { return m_op_attrs; }
                const std::unique_ptr<MKLDNNEmitter>& get_mkldnn_emitter() const
                {
                    return m_mkldnn_emitter;
                }

                const std::string& get_function_name() const { return m_function_name; }
                const std::shared_ptr<ngraph::Function> get_function() { return m_function; }
            protected:
                void compile();

            private:
                void emit_debug_function_entry(codegen::CodeWriter& writer,
                                               Node* node,
                                               const std::vector<TensorViewWrapper>& in,
                                               const std::vector<TensorViewWrapper>& out);
                void emit_debug_function_exit(codegen::CodeWriter& writer,
                                              Node* node,
                                              const std::vector<TensorViewWrapper>& in,
                                              const std::vector<TensorViewWrapper>& out);
                void handle_output_alias(
                    codegen::CodeWriter& writer,
                    const Node&,
                    const std::unordered_map<descriptor::TensorView*, std::vector<size_t>>&);

                bool is_functionally_identical(
                    const Node&,
                    const Node&,
                    const std::unordered_map<const Node*, std::string>& node_cache);
                std::string emit_op_as_function(const Node&, const std::string& function_name);
                std::string strip_comments(const std::string&);
                void release_function() { m_function = nullptr; }
                std::shared_ptr<ngraph::Function> m_function;
                bool m_release_function;
                bool m_is_compiled;
                EntryPoint m_compiled_function;
                std::unique_ptr<codegen::Compiler> m_compiler;
                std::unique_ptr<codegen::ExecutionEngine> m_execution_engine;
                bool m_emit_timing;
                bool m_use_tbb;
                std::unordered_map<std::string, std::string> m_variable_name_map;
                std::map<std::string, size_t> m_name_index_map;

                // Because we are directly accessing the constant data stored in the
                // Constant ops we need to keep a list of shared_ptr to each Constant
                // so they don't get freed before we are done with them
                std::vector<std::shared_ptr<Node>> m_active_constants;

                LayoutDescriptorPtrs parameter_layout_descriptors;
                LayoutDescriptorPtrs result_layout_descriptors;
                std::vector<OpAttributes> m_op_attrs;

                std::unique_ptr<MKLDNNEmitter> m_mkldnn_emitter;

                std::string m_function_name;
            };
        }
    }
}
