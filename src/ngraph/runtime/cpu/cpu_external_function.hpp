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

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"

#endif

#include "ngraph/function.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_emitter.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/state/state.hpp"

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
                                                  codegen::CodeWriter&,
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

            public:
                enum class CPUTensorRole
                {
                    INPUT,
                    CONSTANT,
                    OUTPUT,
                    INTERMEDIATE
                };

                CPU_ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                     bool release_function = true);
                ~CPU_ExternalFunction();
                std::shared_ptr<ngraph::runtime::cpu::CPU_CallFrame> make_call_frame();

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

                size_t add_state(ngraph::State* state)
                {
                    m_states.push_back(state);
                    return m_states.size() - 1;
                }

                const std::string& get_function_name() const { return m_function_name; }
                const std::shared_ptr<ngraph::Function> get_function() { return m_function; }
                // Temporary Memory Pool alignment
                static constexpr size_t s_memory_pool_alignment = 4096;

                std::vector<std::function<void(CPURuntimeContext*)>>& get_functors()
                {
                    return functors;
                }
                std::unordered_map<std::string, void*>& get_tensor_data() { return tensor_data; }
                void*& get_tensor_data(const std::string& name);
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
                std::unordered_map<std::string, std::reference_wrapper<void*>>&
                    get_subgraph_param_ptrs()
                {
                    return subgraph_param_ptrs;
                }
#endif

            protected:
                void build();

#if !defined(NGRAPH_DEX_ONLY)

                void compile();

#endif

                std::vector<ngraph::State*> m_states;

            private:
                // Register passes that are common to codegen and DEX
                void register_common_passes(ngraph::pass::Manager& pass_manager);

                // For non-destructive passthrough kernels, propagate function
                // constant buffers to internal ops
                void propagate_in_place_constant(ngraph::descriptor::Output* output,
                                                 std::string input_name,
                                                 bool dex);
                // For non-destructive passthrough kernels, propagate function
                // input buffers to internal ops
                void propagate_in_place_input(ngraph::descriptor::Output* output,
                                              std::string input_name,
                                              bool dex);
                // For in-place kernels, propagate function output buffers to
                // internal ops
                void propagate_in_place_output(ngraph::descriptor::Output* res_src_output,
                                               std::string output_name,
                                               bool dex);

                // Find in-place concat ops and set appropriate memory pool offset for its arguments
                void process_in_place_concat(std::list<std::shared_ptr<Node>> nodes);

                // For a chain of concat ops, propagate memory pool offsets
                void propagate_in_place_concat(std::shared_ptr<ngraph::op::Concat> concat);
                bool computes_result(Node* node);
                void release_function() { m_function = nullptr; }
#if !defined(NGRAPH_DEX_ONLY)
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

                std::shared_ptr<ngraph::Function> m_function;
                bool m_release_function;
                bool m_emit_timing;

                bool m_use_tbb;
#if !defined(NGRAPH_DEX_ONLY)
                bool m_is_compiled;
#endif
                bool m_direct_execution;
                EntryPoint m_compiled_function;
                std::unordered_map<std::string, std::string> m_variable_name_map;

                std::unordered_map<std::string, CPUTensorRole> m_tensor_roles;

                LayoutDescriptorPtrs parameter_layout_descriptors;
                LayoutDescriptorPtrs result_layout_descriptors;
                std::vector<size_t> m_memory_buffer_sizes;
                std::vector<OpAttributes> m_op_attrs;

                std::unique_ptr<MKLDNNEmitter> m_mkldnn_emitter;

                std::string m_function_name;

                std::vector<std::function<void(CPURuntimeContext*)>> functors;
                std::vector<std::string> op_names;
                std::vector<std::function<bool(CPURuntimeContext*)>> enables;
                std::list<std::pair<std::function<bool(CPURuntimeContext*)>, std::string>>
                    enable_nodename_list;
                std::function<void(CPURuntimeContext*, std::vector<void*>&, std::vector<void*>&)>
                    executor;
                std::unordered_map<std::string, void*> tensor_data;
                std::unordered_map<std::string, bool> tensor_stale;
                std::unordered_map<std::string, std::string> tensor_alias;
                std::list<std::pair<std::reference_wrapper<void*>, size_t>> intermediates_offsets;
                std::list<
                    std::tuple<std::reference_wrapper<void*>, size_t, std::reference_wrapper<bool>>>
                    function_input_index;
                std::list<std::pair<std::reference_wrapper<void*>, size_t>> function_output_index;
                std::unordered_map<std::string, std::shared_ptr<CPU_ExternalFunction>> callees;
                bool m_is_built;
                std::vector<runtime::PerformanceCounter> m_perf_counters;

#if defined(NGRAPH_HALIDE)
                std::unordered_map<std::string, Halide::Func> halide_functions;
                std::unordered_map<std::string, Halide::ImageParam> subgraph_params;
                std::unordered_map<std::string, int> subgraph_param_sizes;
                std::unordered_map<std::string, std::reference_wrapper<void*>> subgraph_param_ptrs;
#endif
            };
        }
    }
}
