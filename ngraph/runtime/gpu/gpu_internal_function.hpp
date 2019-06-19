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
#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

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
            class GPURuntimeConstructor;
            struct GPURuntimeContext;

            class GPUInternalFunction : public GPUCompiledFunction
            {
            public:
                GPUInternalFunction(
                    const std::shared_ptr<ngraph::Function>& function,
                    const std::shared_ptr<GPU_Backend::BackendContext>& shared_context);
                virtual ~GPUInternalFunction();

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
                void build_functions();
                std::string emit_op(EMIT_ARGS);
                std::string
                    compose_manifest(size_t primitive_index,
                                     const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                                     const std::vector<runtime::gpu::GPUTensorWrapper>& out) const;
                void save_manifest_to_disk() const;

                // For non-destructive passthrough kernels, propagate function
                // input buffers to internal ops
                virtual void propagate_in_place_input(ngraph::descriptor::Output* output,
                                                      const std::string& input_name) override;
                // For in-place kernels, propagate function output buffers to
                // internal ops
                virtual void propagate_in_place_output(ngraph::descriptor::Output* res_src_output,
                                                       const std::string& output_name) override;
                std::unordered_map<std::string, std::tuple<TensorRole, size_t, std::string>>
                    m_variable_name_map;
                std::unique_ptr<GPURuntimeConstructor> m_runtime_constructor;
                std::shared_ptr<CodeWriter> m_trace;
                CodeWriter m_manifest;
            };
        }
    }
}
