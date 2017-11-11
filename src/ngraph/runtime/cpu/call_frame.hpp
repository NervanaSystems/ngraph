// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        class PrimaryTensorView;

        namespace cpu
        {
            class CallFrame;

            using EntryPoint_t = void(ngraph::runtime::cpu::CallFrame* call_frame);

            using EntryPoint = std::function<EntryPoint_t>;

            // Compile and execute graphs
            class CallFrame : public ngraph::runtime::CallFrame
            {
            public:
                CallFrame(EntryPoint compiled_function,
                          const std::vector<std::shared_ptr<CallFrame>>& callees);

                /// @brief Invoke the function with values matching the signature of the function.
                ///
                /// Tuples will be expanded into their tensor views to build the call frame.
                void
                    operator()(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& inputs,
                               const std::vector<std::shared_ptr<ngraph::runtime::Value>>& outputs);

                /// @brief Invoke the function with tuples pre-expanded to their underlying
                /// tensor views.
                void tensor_call(const std::vector<std::shared_ptr<TensorView>>& inputs,
                                 const std::vector<std::shared_ptr<TensorView>>& outputs);

                void set_return() { m_return = true; }
                // const std::vector<std::shared_ptr<ngraph::runtime::Value>>& get_inputs();
                // const std::vector<std::shared_ptr<ngraph::runtime::Value>>& get_outputs();

                void* get_input_data(size_t index);
                void* get_output_data(size_t index);

            protected:
                bool m_return;
                EntryPoint m_compiled_function;
                std::vector<std::shared_ptr<CallFrame>> m_callees;

                std::vector<void*> m_inputs;
                std::vector<void*> m_outputs;
            };
        }
    }
}
