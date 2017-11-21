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

        namespace ngvm
        {
            class Instruction;

            // A VM for executing lightly-compiled graph functions.
            class CallFrame : public ngraph::runtime::CallFrame
            {
            public:
                CallFrame(
                    size_t n_inputs,
                    size_t n_outputs,
                    size_t frame_size,
                    const TensorViewPtrs& temps,
                    size_t initial_pc,
                    const std::shared_ptr<std::vector<std::shared_ptr<Instruction>>>& instructions);

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
                    return dynamic_cast<ParameterizedTensorView<ET>*>(m_tensor_views[i].get());
                }

                template <typename ET>
                typename ET::type* get_tensor_view_data(size_t i)
                {
                    return static_cast<typename ET::type*>(
                        get_parameterized_tensor_view<ET>(i)->get_data_ptr());
                }

            protected:
                size_t m_n_inputs;
                size_t m_n_outputs;
                size_t m_frame_size;
                TensorViewPtrs m_tensor_views;
                size_t m_initial_pc;
                std::shared_ptr<std::vector<std::shared_ptr<Instruction>>> m_instructions;
                size_t m_pc;
                size_t m_next_pc;
                bool m_return;
            };
        }
    }
}
