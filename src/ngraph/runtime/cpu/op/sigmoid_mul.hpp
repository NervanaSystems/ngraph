/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/util.hpp"

#include <array>

namespace ngraph
{
    namespace op
    {
        /// \brief Fused Sigmoid functions (logistic and tanh) with multiplication forward prop.
        class SigmoidMultiply : public util::RequiresTensorViewArgs
        {
        public:
            /// Defines valid function types
            enum class FunctionType
            {
                Logistic,
                Tanh,
                Identity,
                NumTypes
            };
            /// Input nodes are expected to be actual inputs where the corresponding input
            /// FunctionType will be applied to those inputs in the fused operation.
            SigmoidMultiply(std::shared_ptr<Node> input_0,
                            std::shared_ptr<Node> input_1,
                            const FunctionType input_0_type,
                            const FunctionType input_1_type);
            /// WARNING: copy_with_new_args() implicitly expects new args must match the original input function types.
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            FunctionType get_input_func_type(const unsigned int index) const
            {
                return m_input_type[index];
            }
            /// Identifies the corresponding FunctionType for the input node.
            static FunctionType identify_node_type(const std::shared_ptr<ngraph::Node>& node);

        private:
            std::array<FunctionType, 2> m_input_type;
        };

        /// \brief Elementwise SigmoidMultiplyBackprop operation.
        ///
        class SigmoidMultiplyBackprop : public util::RequiresTensorViewArgs
        {
        public:
            typedef SigmoidMultiply::FunctionType FunctionType;
            /// \brief Constructs a SigmoidMultiplyBackprop operation.
            ///
            /// \param input_0 Forward input node 0.
            /// \param input_1 Forward input node 1.
            /// \param delta Backprop delta node.
            /// \param input_type Function type for the input nodes.
            SigmoidMultiplyBackprop(std::shared_ptr<Node> input_0,
                                    std::shared_ptr<Node> input_1,
                                    std::shared_ptr<ngraph::Node> delta,
                                    const std::array<FunctionType, 2>& input_type);
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
            FunctionType get_input_func_type(const unsigned int index) const
            {
                return m_input_type[index];
            }

        private:
            std::array<FunctionType, 2> m_input_type;
        };
    }
}
