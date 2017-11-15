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

#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Zero-padding operation
        ///
        /// ## Parameters
        ///
        /// |                  | Description                                                                                              |
        /// | ---------------- | -------------------------------------------------------------------------------------------------------- |
        /// | `padding_before` | The per-axis size of padding to add below the input's zero-positions. Size must match rank of the input. |
        /// | `padding_after`  | The per-axis size of padding to add above the input's max-positions. Size much match rank of the input.  |
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                             |
        /// | ----- | --------------------------------- | --------------------------------------- |
        /// | `arg` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any element type and shape. |
        ///
        /// ## Output
        ///
        /// | Type                                                                                                                                         | Description                                                                                                                                                                                                         |
        /// | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[d_1 + \mathtt{padding\_before}[0] + \mathtt{padding\_after}[0],\dots,d_n + \mathtt{padding\_before}[n-1] + \mathtt{padding\_after}[n-1]]\f$ | The tensor \f$T\f$ where \f$T[i_1,\dots,i_n] = \mathtt{arg}[i_1 - \mathtt{padding\_before}[0],\dots,i_n - \mathtt{padding\_before}[n-1]]\f$ if for every \f$i_k\f$, \f$\mathtt{padding\_before}[k-1] \le i_k \lt \mathtt{padding\_after}[k-1] + d_k\f$, else \f$0\f$. |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status           |
        /// | ------- | ---------------- |
        /// | NGVM    | Not implemented. |

        class PadZero : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a zero-padding operation.
            ///
            /// \param padding_before The per-axis size of padding to add below the input's zero-positions. Size must match rank of the input.
            /// \param padding_after  The per-axis size of padding to add above the input's max-positions. Size much match rank of the input.
            /// \param arg            Node that produces the input tensor to be padded.
            PadZero(const Shape& padding_before,
                    const Shape& padding_after,
                    const std::shared_ptr<Node>& arg);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 1)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<PadZero>(m_padding_before, m_padding_after, new_args.at(0));
            }

            const Shape& get_padding_before() const { return m_padding_before; }
            const Shape& get_padding_after() const { return m_padding_after; }
        protected:
            Shape m_padding_before;
            Shape m_padding_after;

            /*        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;*/
        };
    }
}
