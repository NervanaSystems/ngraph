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
            class CPU_CallFrame;
            class CPU_ExternalFunction;

            class PerformanceCounter
            {
            public:
                PerformanceCounter(const char* n, size_t us, size_t calls)
                    : m_name(n)
                    , m_total_microseconds(us)
                    , m_call_count(calls)
                {
                }
                const std::string& name() const { return m_name; }
                size_t total_microseconds() const { return m_total_microseconds; }
                size_t microseconds() const { return m_total_microseconds / m_call_count; }
                size_t call_count() const { return m_call_count; }
            private:
                std::string m_name;
                size_t m_total_microseconds;
                size_t m_call_count;
            };

            using EntryPoint_t = void(void** inputs, void** outputs);

            using EntryPoint = std::function<EntryPoint_t>;

            // Compile and execute graphs
            class CPU_CallFrame : public ngraph::runtime::CallFrame
            {
            public:
                CPU_CallFrame(std::shared_ptr<CPU_ExternalFunction> external_function,
                              EntryPoint compiled_function);

                /// @brief Invoke the function with values matching the signature of the function.
                ///
                /// Tuples will be expanded into their tensor views to build the call frame.
                void call(const std::vector<std::shared_ptr<runtime::TensorView>>& inputs,
                          const std::vector<std::shared_ptr<runtime::TensorView>>& outputs);

                /// @brief Invoke the function with tuples pre-expanded to their underlying
                /// tensor views.
                void tensor_call(const std::vector<std::shared_ptr<TensorView>>& inputs,
                                 const std::vector<std::shared_ptr<TensorView>>& outputs);

                std::vector<ngraph::runtime::cpu::PerformanceCounter> get_performance_data() const;

            protected:
                std::shared_ptr<CPU_ExternalFunction> m_external_function;
                EntryPoint m_compiled_function;
            };
        }
    }
}
