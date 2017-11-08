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
        /// \brief Batched convolution operation
        ///
        /// FIXME: grandpa, what's convolution
        ///
        /// ## Inputs
        ///
        /// |        | Type                                                                                                    | Description                                                                        |
        /// | ------ | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
        /// | `arg0` | \f$N[d_\textit{imgs},d_\textit{ci},d_1,\dots,d_n]~(n \geq 0)\f$                                         | (Images) A tensor of rank>2, any shape, and any numeric element type.              |
        /// | `arg1` | \f$N[d_\textit{co},d_\textit{ci},d^k_1,\dots,d^k_n]~(n \geq 0, 0 < d_\textit{i} \le d^k_\textit{i})\f$  | (Convolution kernels) A tensor with the same rank and element type as `arg0`, with |
        /// |        |                                                                                                         | the second dimension (corresponding to input channels) matching `arg0`, and with   |
        /// |        |                                                                                                         | the subsequent dimensions (corresponding to the convolution kernels) no greater    |
        /// |        |                                                                                                         | than the corresponding image dimension.                                            |
        ///
        /// ## Output
        ///
        /// | Type                                                                                        | Description                                                                     |
        /// | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
        /// | \f$N[d_\textit{imgs},d_\textit{co},d'_1,\dots,d'_n]\f$ where \f$d'_i = d_i - (d^k_i - 1)\f$ | The result of convolution (FIXME: vague)                                        |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status           |
        /// | ------- | ---------------- |
        /// | NGVM    | Not implemented. |

        class Convolution : public Builtin
        {
        public:
            /// \brief Constructs a convolution operation.
            ///
            /// \param arg0           Node that produces the input tensor for the input images.
            /// \param arg1           Node that produces the input tensor for the convolution kernels.
            Convolution(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : Builtin({arg0, arg1})
            {
            }

            virtual std::string description() const override { return "Convolution"; }
            virtual void propagate_types() override;
            /*        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;*/
        };
    }
}
