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
            class GPU_CallFrame;

            class GPU_Backend : public Backend
            {
            public:
                std::shared_ptr<ngraph::runtime::gpu::GPU_CallFrame> make_call_frame(
                    const std::shared_ptr<ngraph::runtime::gpu::GPU_ExternalFunction>&
                        external_function);

                std::shared_ptr<ngraph::runtime::TensorView>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                std::shared_ptr<ngraph::runtime::TensorView>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape) override;

                bool compile(std::shared_ptr<Function> func) override;

                bool call(std::shared_ptr<Function> func,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) override;

            private:
                class FunctionInstance
                {
                public:
                    std::shared_ptr<GPU_ExternalFunction> m_external_function;
                    std::shared_ptr<GPU_CallFrame> m_call_frame;
                };

                std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;
            };
        }
    }
}
