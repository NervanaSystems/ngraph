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
        namespace cpu
        {
            class CPU_ExternalFunction;
            class CPU_CallFrame;

            class CPUExecutable : public runtime::Executable
            {
                friend class CPU_Backend;

            public:
                bool execute(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                             const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

                std::shared_ptr<CPU_CallFrame> get_call_frame();

                std::vector<PerformanceCounter> get_performance_data() const override;

            private:
                CPUExecutable(std::shared_ptr<Function> function,
                              bool enable_performance_collection);

                std::shared_ptr<CPU_ExternalFunction> m_external_function;
                std::shared_ptr<CPU_CallFrame> m_call_frame;
                bool m_performance_counters_enabled = false;
            };
        }
    }
}
