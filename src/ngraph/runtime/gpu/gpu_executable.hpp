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

#include <map>
#include <memory>

#include "gpu_backend_visibility.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPUPrimitiveEmitter;
            struct GPURuntimeContext;
            class CudaContextManager;

            using EntryPoint_t = void(void** inputs, void** outputs, GPURuntimeContext* ctx);
            using EntryPoint = std::function<EntryPoint_t>;

            class GPUExecutable : public Executable
            {
            public:
                GPUExecutable(std::shared_ptr<Function> func, bool enable_timing);

                bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

                // void remove_compiled_function(std::shared_ptr<Function> func) override;
                std::vector<PerformanceCounter> get_performance_data() const override;

            private:
                std::shared_ptr<GPUCompiledFunction> m_compiled_function;
                bool m_performance_counters_enabled = false;
                EntryPoint m_runtime;
                std::vector<void*> m_inputs;
                std::vector<void*> m_outputs;

                /// \brief Convert a vector of Tensor into a vector of void* where each void*
                /// points to a Tensor's data buffer.
                /// \param target Pointer to a pre-allocated array of void* with
                /// size >= source.size()
                /// \param source Source vector of Tensors
                static void
                    initialize_io(void** target,
                                  const std::vector<std::shared_ptr<runtime::Tensor>>& source);

                std::shared_ptr<GPUBackend::BackendContext> m_context;
            };
        }
    }
}
