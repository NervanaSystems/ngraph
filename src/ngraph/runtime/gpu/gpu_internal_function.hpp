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

#include "ngraph/function.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"
#include "ngraph/runtime/gpu/gpu_compiled_function.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter;
            struct GPURuntimeContext;

            class GPU_InternalFunction : public GPU_CompiledFunction
            {
            public:
                GPU_InternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                     std::shared_ptr<GPU_Backend::BackendContext>& shared_context);
                virtual ~GPU_InternalFunction();

                virtual std::string add_to_runtime(size_t primitive_index,
                                            const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                                            const std::vector<runtime::gpu::GPUTensorWrapper>& out) override;
                virtual void compile() override;
                virtual void get_performance_data(std::vector<runtime::PerformanceCounter>& rc) const override;
            private:

                // For non-destructive passthrough kernels, propagate function
                // input buffers to internal ops
                virtual void propagate_in_place_input(ngraph::descriptor::Output* output,
                                                      std::string input_name) override;
                // For in-place kernels, propagate function output buffers to
                // internal ops
                virtual void propagate_in_place_output(ngraph::descriptor::Output* res_src_output,
                                                       std::string output_name) override;
                std::map<std::string, size_t> m_name_index_map;
                std::unordered_map<std::string, std::string> m_variable_name_map;
                std::unordered_map<Node*, Node*> m_node_function_map;
            };
        }
    }
}
