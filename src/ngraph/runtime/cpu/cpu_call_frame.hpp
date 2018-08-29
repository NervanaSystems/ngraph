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
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_CallFrame;
            class CPU_ExternalFunction;

            using EntryPoint_t = void(void** inputs, void** outputs, CPURuntimeContext* ctx);

            using EntryPoint = std::function<EntryPoint_t>;

            // Compile and execute graphs
            class CPU_CallFrame
            {
            public:
                CPU_CallFrame(std::shared_ptr<CPU_ExternalFunction> external_function,
                              EntryPoint compiled_function);
                ~CPU_CallFrame();

                /// \brief Invoke the function with values matching the signature of the function.
                ///
                /// Tuples will be expanded into their tensor views to build the call frame.
                void call(const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& inputs);

                void propagate_layouts(const std::vector<std::shared_ptr<runtime::TensorView>>& tvs,
                                       const LayoutDescriptorPtrs& layouts) const;

                void setup_runtime_context();
                void cleanup_runtime_context();

            protected:
                std::shared_ptr<CPU_ExternalFunction> m_external_function;
                EntryPoint m_compiled_function;
                CPURuntimeContext* ctx;
            };
        }
    }
}
