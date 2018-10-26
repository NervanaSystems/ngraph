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
#include <string>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_Debugger
            {
            public:
                CPU_Debugger(CPU_CallFrame& callframe);
                ~CPU_Debugger();

                /// \brief Invoke the function with values matching the signature of the function.
                ///
                /// Tuples will be expanded into their tensor views to build the call frame.
                void call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

                /// \brief Execute a single operation
                bool step();

                /// \brief Continue to execute from the current PC
                void resume();

                /// \brief Add a breakpoint to a node
                bool add_breakpoint(std::shared_ptr<Node> op);
                /// \brief Remove a breakpoint from a node
                bool delete_breakpoint(std::shared_ptr<Node> op);

                void* inspect(std::shared_ptr<Node> op, size_t output_index = 0);

            protected:
                CPU_Debugger(const CPU_Debugger&) = delete;
                CPU_Debugger(CPU_Debugger&&) = delete;
                CPU_Debugger& operator=(const CPU_Debugger&) = delete;

                CPU_CallFrame& m_callframe;
                std::vector<std::shared_ptr<runtime::Tensor>> m_inputs;
                std::vector<std::shared_ptr<runtime::Tensor>> m_outputs;
            };
        }
    }
}
