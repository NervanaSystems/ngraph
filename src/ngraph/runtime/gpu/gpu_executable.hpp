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

#include <map>
#include <memory>

#include "ngraph/runtime/executable.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            static size_t alignment = 64;

            class GPU_ExternalFunction;
            class GPUPrimitiveEmitter;
            struct GPURuntimeContext;
            class CudaContextManager;
            class BackendContext;

            using EntryPoint_t = void(void** inputs, void** outputs, GPURuntimeContext* ctx);
            using EntryPoint = std::function<EntryPoint_t>;

            class GPUExecutable : public runtime::Executable
            {
                friend class GPU_Backend;

            public:
                bool execute(const std::vector<runtime::Tensor*>& outputs,
                             const std::vector<runtime::Tensor*>& inputs) override;

                std::vector<PerformanceCounter> get_performance_data() const override;

            private:
                std::shared_ptr<GPU_ExternalFunction> m_external_function;
                bool m_performance_counters_enabled = false;
                EntryPoint m_compiled_function;
                std::vector<void*> m_inputs;
                std::vector<void*> m_outputs;
                std::shared_ptr<BackendContext> m_context;

                GPUExecutable(std::shared_ptr<BackendContext> context,
                              std::shared_ptr<Function> function,
                              bool enable_performance_collection);

                /// \brief Convert a vector of Tensor into a vector of void* where each void*
                /// points to a Tensor's data buffer.
                /// \param target Pointer to a pre-allocated array of void* with
                /// size >= source.size()
                /// \param source Source vector of Tensors
                static void initialize_io(void** target,
                                          const std::vector<runtime::Tensor*>& source);
            };
        }
    }
}
