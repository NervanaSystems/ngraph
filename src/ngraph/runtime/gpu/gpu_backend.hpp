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

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            static size_t alignment = 64;

            class GPUCompiledFunction;
            class GPUPrimitiveEmitter;
            struct GPURuntimeContext;
            class CudaContextManager;
            class GPUExecutable;

            using EntryPoint_t = void(void** inputs, void** outputs, GPURuntimeContext* ctx);
            using EntryPoint = std::function<EntryPoint_t>;

            BackendConstructor GPU_BACKEND_API get_backend_constructor_pointer();
            class GPUBackend : public Backend
            {
            public:
                GPUBackend();

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape) override;

                std::shared_ptr<runtime::Executable> compile(std::shared_ptr<Function> func,
                                                             bool timing_enabled = false) override;

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
                std::map<std::shared_ptr<Function>, std::shared_ptr<Executable>> m_exec_map;
            };
        }
    }
}
