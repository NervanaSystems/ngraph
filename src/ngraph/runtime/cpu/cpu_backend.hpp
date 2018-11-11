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
#include <vector>

#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class CPU_CallFrame;

            class CPU_Backend : public runtime::Backend
            {
            public:
                std::shared_ptr<CPU_CallFrame>
                    make_call_frame(const std::shared_ptr<CPU_ExternalFunction>& external_function);

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape,
                                  void* memory_pointer) override;

                std::shared_ptr<ngraph::runtime::Tensor>
                    create_tensor(const ngraph::element::Type& element_type,
                                  const Shape& shape) override;

                runtime::Handle compile(const std::shared_ptr<Function>& func) override;

                bool call(runtime::Handle handle,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

                void remove_compiled_function(runtime::Handle handle) override;
                std::shared_ptr<CPU_CallFrame> get_call_frame(runtime::Handle handle);

                void enable_performance_data(runtime::Handle handle, bool enable) override;
                std::vector<PerformanceCounter>
                    get_performance_data(runtime::Handle handle) const override;

            private:
                class FunctionInstance
                {
                public:
                    std::shared_ptr<CPU_ExternalFunction> m_external_function;
                    std::shared_ptr<CPU_CallFrame> m_call_frame;
                    bool m_performance_counters_enabled = false;
                };

                std::vector<std::shared_ptr<FunctionInstance>> m_instances;
            };
        }
    }
}
