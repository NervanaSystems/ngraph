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
#include <mutex>

#include "cpu_backend_visibility.h"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/runtime/allocator.hpp"
#include "ngraph/runtime/executable.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class CPU_CallFrame;

            class CPU_BACKEND_API CPU_Executable : public runtime::Executable
            {
            public:
                CPU_Executable(std::shared_ptr<Function> func,
                               ngraph::pass::PassConfig& pass_config,
                               Allocator* allocator,
                               bool performance_counters_enabled);
                bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

                std::shared_ptr<CPU_CallFrame> get_call_frame();

                std::vector<PerformanceCounter> get_performance_data() const override;

                std::shared_ptr<runtime::Tensor> create_input_tensor(size_t input_index) override;

                std::shared_ptr<runtime::Tensor> create_input_tensor(size_t input_index,
                                                                     void* memory_pointer) override;

                std::shared_ptr<runtime::Tensor> create_output_tensor(size_t output_index) override;

                std::shared_ptr<runtime::Tensor>
                    create_output_tensor(size_t output_index, void* memory_pointer) override;

                std::vector<std::shared_ptr<runtime::Tensor>>
                    create_input_tensor(size_t input_index,
                                        size_t pipeline_depth,
                                        std::vector<void*> memory_pointers) override;

                std::vector<std::shared_ptr<runtime::Tensor>>
                    create_input_tensor(size_t input_index, size_t pipeline_depth) override;

                std::vector<std::shared_ptr<runtime::Tensor>>
                    create_output_tensor(size_t output_index, size_t pipeline_depth) override;

                std::vector<std::shared_ptr<runtime::Tensor>>
                    create_output_tensor(size_t output_index,
                                         size_t pipeline_depth,
                                         std::vector<void*> memory_pointers) override;

            private:
                std::shared_ptr<ngraph::op::Parameter> get_parameter(size_t index) const;
                std::shared_ptr<ngraph::op::Result> get_result(size_t index) const;
                class FunctionInstance
                {
                public:
                    std::shared_ptr<CPU_ExternalFunction> m_external_function = nullptr;
                    std::shared_ptr<CPU_CallFrame> m_call_frame = nullptr;
                    bool m_performance_counters_enabled = false;
                } m_function_instance;
            };
        }
    }
}
