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
#include "ngraph/runtime/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        class PrimaryTensorView;

        // A VM for executing lightly-compiled graph functions.
        class CallFrame
        {
        public:
            CallFrame(
                size_t                                                            n_inputs,
                size_t                                                            n_outputs,
                const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>&  temps,
                size_t                                                            initial_pc,
                const std::shared_ptr<std::vector<std::shared_ptr<Instruction>>>& instructions);

            /// @brief Invoke the function with values matching the signature of the function.
            ///
            /// Tuples will be expanded into their tensor views to build the call frame.
            void operator()(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& inputs,
                            const std::vector<std::shared_ptr<ngraph::runtime::Value>>& outpus);

            /// @brief Invoke the function with tuples pre-expanded to their underlying tensor views.
            void tensor_call(
                const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& inputs,
                const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& outpus);

            void set_return() { m_return = true; }

            std::shared_ptr<TensorView> get_tensor(size_t i) { return m_tensors[i]; }

            template <typename ET>
            ParameterizedTensorView<ET>* get_parameterized_tensor(size_t i)
            {
                return m_tensors[i]->get_parameterized_tensor<ET>();
            }

        protected:
            size_t                                                     m_n_inputs;
            size_t                                                     m_n_outputs;
            std::vector<std::shared_ptr<ngraph::runtime::TensorView>>  m_tensors;
            size_t                                                     m_initial_pc;
            std::shared_ptr<std::vector<std::shared_ptr<Instruction>>> m_instructions;
            size_t                                                     m_pc;
            size_t                                                     m_next_pc;
            bool                                                       m_return;
        };
    }
}
