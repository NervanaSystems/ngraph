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

#pragma once

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(NGRAPH_HALIDE)
#include <Halide.h>
#endif

#if !defined(NGRAPH_DEX_ONLY)

#include "ngraph/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"

#endif

#include "ngraph/function.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_emitter.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/state/state.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class CPU_Emitter;
            class CPU_CallFrame;
            class CPU_Debugger;

#if !defined(NGRAPH_DEX_ONLY)

            using OpFunction = std::function<void(CPU_ExternalFunction* external_function,
                                                  CodeWriter&,
                                                  const ngraph::Node*,
                                                  const std::vector<TensorViewWrapper>& inputs,
                                                  const std::vector<TensorViewWrapper>& outputs)>;

            using OpMap = std::unordered_map<std::type_index, OpFunction>;
#endif

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
                friend class CPU_CallFrame;
                friend class CPU_Debugger;
                friend class CPU_Executable;

            public:
                CPU_ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                     bool release_function = true);
                ~CPU_ExternalFunction();
                std::shared_ptr<ngraph::runtime::cpu::CPU_CallFrame>
                    make_call_frame(ngraph::pass::PassConfig& pass_config);

                const LayoutDescriptorPtrs& get_parameter_layout_descriptors();
                const LayoutDescriptorPtrs& get_result_layout_descriptors();
                const std::vector<size_t>& get_memory_buffer_sizes() const
                {
                    return m_memory_buffer_sizes;
                }
                const std::vector<OpAttributes>& get_op_attrs() const { return m_op_attrs; }
                const std::unique_ptr<MKLDNNEmitter>& get_mkldnn_emitter() const
                {
                    return m_mkldnn_emitter;
                }

                // Return the tuple including the string to create mkldnn primitive, the deps and the index in CODEGEN
                const std::tuple<std::string, std::vector<size_t>, size_t>&
                    get_primitive_build_tuple(const Node* node) const
                {
                    auto it = m_node_primitive_string_deps_index_map.find(node);
                    NGRAPH_CHECK(it != m_node_primitive_string_deps_index_map.end(),
                                 "Primitive build tuple not found for node ",
                                 node->description());

                    return it->second;
                }

                size_t add_state(ngraph::State* state)
                {
                    m_states.push_back(state);
                    return m_states.size() - 1;
                }

                const std::string& get_function_name() const { return m_function_name; }
                const std::shared_ptr<ngraph::Function> get_function() { return m_function; }
                // Temporary Memory Pool alignment
                static constexpr size_t s_memory_pool_alignment = 4096;

                std::vector<CPUKernelFunctor>& get_functors() { return functors; }
                // return an index into the cpu_runtime_context's buffer_data vector to get the tensor
                size_t get_buffer_index(const std::string& name);
                size_t get_buffer_size() const { return m_buffer_size; }
                std::function<void(CPURuntimeContext*, std::vector<void*>&, std::vector<void*>&)>&
                    get_executor()
                {
                    return executor;
                }
                std::unordered_map<std::string, std::shared_ptr<CPU_ExternalFunction>>&
                    get_callees()
                {
                    return callees;
                }
                bool is_direct_execution() const { return m_direct_execution; }
                void write_to_file(const std::string& code,
                                   const std::string& directory,
                                   const std::string& filename);

                const std::vector<PerformanceCounter>& get_perf_counters();

#if defined(NGRAPH_HALIDE)
                std::unordered_map<std::string, Halide::Func>& get_halide_functions()
                {
                    return halide_functions;
                }
                std::unordered_map<std::string, Halide::ImageParam>& get_subgraph_params()
                {
                    return subgraph_params;
                }
                std::unordered_map<std::string, int>& get_subgraph_param_sizes()
                {
                    return subgraph_param_sizes;
                }
                std::unordered_map<std::string, size_t>> &get_subgraph_param_indices()
                {
                    return subgraph_param_indices;
                }
#endif

            protected:
                void build(ngraph::pass::PassConfig& pass_config);

#if !defined(NGRAPH_DEX_ONLY)

                void compile(ngraph::pass::PassConfig& pass_config);

#endif

                std::vector<ngraph::State*> m_states;

            private:
                // Register passes that are common to codegen and DEX
                void register_common_passes(ngraph::pass::Manager& pass_manager,
                                            ngraph::pass::PassConfig& pass_config);

                bool computes_result(Node* node);
                void release_function() { m_function = nullptr; }
#if !defined(NGRAPH_DEX_ONLY)
                void emit_debug_function_entry(CodeWriter& writer,
                                               Node* node,
                                               const std::vector<TensorViewWrapper>& in,
                                               const std::vector<TensorViewWrapper>& out);
                void emit_debug_function_exit(CodeWriter& writer,
                                              Node* node,
                                              const std::vector<TensorViewWrapper>& in,
                                              const std::vector<TensorViewWrapper>& out);
                void handle_output_alias(
                    CodeWriter& writer,
                    const Node&,
                    const std::unordered_map<descriptor::Tensor*, std::vector<size_t>>&);

                bool is_functionally_identical(
                    const Node&,
                    const Node&,
                    const std::unordered_map<const Node*, std::string>& node_cache);
                std::string emit_op_as_function(const Node&, const std::string& function_name);
                std::string strip_comments(const std::string&);

                std::unique_ptr<codegen::Compiler> m_compiler;
                std::unique_ptr<codegen::ExecutionEngine> m_execution_engine;

                std::map<std::string, size_t> m_name_index_map;

                // Because we are directly accessing the constant data stored in the
                // Constant ops we need to keep a list of shared_ptr to each Constant
                // so they don't get freed before we are done with them
                std::vector<std::shared_ptr<Node>> m_active_constants;
#endif
                static bool is_codegen(const ngraph::pass::PassConfig& pc);
                std::unordered_set<descriptor::Tensor*>&
                    get_tensor_set(descriptor::Tensor* output_tensor);

                std::shared_ptr<ngraph::Function> m_function;
                bool m_release_function;
                bool m_emit_timing;

                bool m_use_tbb;
#if !defined(NGRAPH_DEX_ONLY)
                bool m_is_compiled;
#endif
                bool m_direct_execution;

                /// Function that initializes the context used in codegen mode.
                InitContextFuncCG m_compiled_init_ctx_func;

                /// Function that destroys the context used in codegen mode.
                DestroyContextFuncCG m_compiled_destroy_ctx_func;

                EntryPoint m_compiled_function;
                std::unordered_map<std::string, std::string> m_variable_name_map;
                std::unordered_map<std::string, std::pair<std::size_t, std::size_t>>
                    m_variable_input_index_offset_map;
                std::unordered_map<std::string, std::pair<std::size_t, std::size_t>>
                    m_variable_output_index_offset_map;

                std::unordered_map<std::string, ngraph::TensorRole> m_tensor_roles;

                LayoutDescriptorPtrs parameter_layout_descriptors;
                LayoutDescriptorPtrs result_layout_descriptors;
                std::vector<size_t> m_memory_buffer_sizes;
                std::vector<OpAttributes> m_op_attrs;

                std::unique_ptr<MKLDNNEmitter> m_mkldnn_emitter;

                std::string m_function_name;

                std::vector<CPUKernelFunctor> functors;
                std::vector<std::string> op_names;
                std::vector<std::function<bool(CPURuntimeContext*)>> enables;
                std::list<std::pair<std::function<bool(CPURuntimeContext*)>, std::string>>
                    enable_nodename_list;
                std::function<void(CPURuntimeContext*, std::vector<void*>&, std::vector<void*>&)>
                    executor;
                // name of a tensor and index into the cpu_runtime_context's buffer_data vector to get the tensor
                std::unordered_map<std::string, size_t> m_buffer_indices;
                std::unordered_map<std::string, bool> tensor_stale;
                // Each tensor is put into one buffer set.
                // All the tensors in the same buffer set share the same memory buffer.
                // bufferID_to_tensorSets maps bufferID to the pair of TensorRole and buffer set.
                // TensorRole is INPUT, CONSTANT, OUTPUT, or INTERMEDIATE,
                // which tells from where the memory buffer comes.
                std::unordered_map<
                    size_t,
                    std::pair<ngraph::TensorRole, std::unordered_set<descriptor::Tensor*>>>
                    bufferID_to_tensorSets;
                // tensor_to_bufferID maps tensor to the ID of the buffer set it belongs to.
                std::unordered_map<descriptor::Tensor*, size_t> tensor_to_bufferID;
                std::unordered_map<std::string, std::string> tensor_alias;

                // index into the cpu_runtime_context's buffer_data vector to get a tensor,
                // and the tensor's offset into the memory allocated for intermediates.
                // used to calculate the correct address at runtime
                std::list<std::pair<size_t, size_t>> intermediates_offsets;
                // index into the cpu_runtime_context's buffer_data vector to get a tensor,
                // and the tensor pointer.
                // used to get the address at runtime
                std::list<std::pair<size_t, void*>> constant_tensor_data;
                // index into the cpu_runtime_context's buffer_data vector to get a tensor,
                // input index, offset into the input, and if the input is stale
                // used to calculate the correct address at runtime
                std::list<std::tuple<size_t, size_t, size_t, std::reference_wrapper<bool>>>
                    function_input_index_offset;
                // index to the cpu_runtime_context's buffer_data vector to get a tensor,
                // output index, and offset into the output.
                // used to calculate the correct address at runtime
                std::list<std::tuple<size_t, size_t, size_t>> function_output_index_offset;
                //size of the cpu_runtime_context's buffer_data vector.
                size_t m_buffer_size = 0;
                std::unordered_map<std::string, std::shared_ptr<CPU_ExternalFunction>> callees;
                bool m_is_built;
                std::vector<runtime::PerformanceCounter> m_perf_counters;

#if defined(NGRAPH_HALIDE)
                std::unordered_map<std::string, Halide::Func> halide_functions;
                std::unordered_map<std::string, Halide::ImageParam> subgraph_params;
                std::unordered_map<std::string, int> subgraph_param_sizes;
                std::unordered_map<std::string, size_t> subgraph_param_indices;
#endif

                /// Map each node with mkldnn implementation to its mkldnn primitive creating string, deps, and mkldnn primitive index.
                std::map<const Node*, std::tuple<std::string, std::vector<size_t>, size_t>>
                    m_node_primitive_string_deps_index_map;
                /// Name of the file to store descriptors for mkldnn_primitives
                const std::string m_desc_filename = "desc_file";
            };
        }
    }
}
