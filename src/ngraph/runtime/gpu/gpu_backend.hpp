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

#include <map>
#include <memory>

#include "ngraph/runtime/backend.hpp"

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

            using EntryPoint_t = void(void** inputs, void** outputs, GPURuntimeContext* ctx);
            using EntryPoint = std::function<EntryPoint_t>;

            class GPU_Backend : public Backend
            {
            public:
                GPU_Backend();

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape) override;

                Handle compile(std::shared_ptr<Function> func) override;

                bool call(std::shared_ptr<Function> func,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

                void remove_compiled_function(std::shared_ptr<Function> func) override;
                void enable_performance_data(std::shared_ptr<Function> func, bool enable) override;
                std::vector<PerformanceCounter>
                    get_performance_data(std::shared_ptr<Function> func) const override;

                bool is_supported(const Node& node) const override;

                class BackendContext
                {
                public:
                    BackendContext();
                    ~BackendContext();
                    void prepare_runtime_context();
                    void bind_cuda_context_to_thread();

                    std::unique_ptr<GPURuntimeContext> m_runtime_context;
                    std::unique_ptr<GPUPrimitiveEmitter> m_primitive_emitter;

                private:
                    std::unique_ptr<CudaContextManager> m_cuda_manager;
                };

            private:
                class FunctionInstance
                {
                public:
                    std::shared_ptr<GPU_ExternalFunction> m_external_function;
                    bool m_performance_counters_enabled = false;
                    EntryPoint m_compiled_function;
                    std::vector<void*> m_inputs;
                    std::vector<void*> m_outputs;
                };

                /// \brief Convert a vector of Tensor into a vector of void* where each void*
                /// points to a Tensor's data buffer.
                /// \param target Pointer to a pre-allocated array of void* with
                /// size >= source.size()
                /// \param source Source vector of Tensors
                static void
                    initialize_io(void** target,
                                  const std::vector<std::shared_ptr<runtime::Tensor>>& source);

                std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;
                std::shared_ptr<BackendContext> m_context;
            };
        }
    }
}
