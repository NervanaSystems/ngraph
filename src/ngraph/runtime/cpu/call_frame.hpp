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

            using EntryPoint = std::function<void(ngraph::runtime::cpu::CallFrame*,
                                                  ngraph::runtime::TensorViewPtrs&,
                                                  const std::vector<std::shared_ptr<CallFrame>>&)>;

            // Compile and execute graphs
            class CallFrame : public ngraph::runtime::CallFrame
            {
            public:
                CallFrame(EntryPoint compiled_function,
                          size_t n_inputs,
                          size_t n_outputs,
                          const TensorViewPtrs& temps,
                          const std::vector<std::shared_ptr<CallFrame>>& callees);

                /// @brief Invoke the function with values matching the signature of the function.
                ///
                /// Tuples will be expanded into their tensor views to build the call frame.
                void
                    operator()(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& inputs,
                               const std::vector<std::shared_ptr<ngraph::runtime::Value>>& outputs);

                /// @brief Invoke the function with tuples pre-expanded to their underlying tensor views.
                void tensor_call(const TensorViewPtrs& inputs, const TensorViewPtrs& outputs);

                void set_return() { m_return = true; }
                std::shared_ptr<TensorView> get_tensor_view(size_t i) { return m_tensor_views[i]; }
                template <typename ET>
                ParameterizedTensorView<ET>* get_parameterized_tensor_view(size_t i)
                {
                    return m_tensor_views[i]->get_parameterized_tensor_view<ET>();
                }

                template <typename ET>
                typename ET::type* get_tensor_view_data(size_t i)
                {
                    return &get_parameterized_tensor_view<ET>(i)->get_vector()[0];
                }

            protected:
                size_t m_n_inputs;
                size_t m_n_outputs;
                TensorViewPtrs m_tensor_views;
                bool m_return;
                EntryPoint m_compiled_function;
                std::vector<std::shared_ptr<CallFrame>> m_callees;
            };
        }
    }
}
