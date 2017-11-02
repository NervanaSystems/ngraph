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
        /// \brief Elementwise sign operation.
        ///
        /// Maps each element of the input tensor to -1 (if it is negative), 0 (if it is zero), or 1 (if it is positive).
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                     |
        /// | ----- | --------------------------------- | ----------------------------------------------- |
        /// | `arg` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                          |
        /// | ---------------------- | ------------------------------------------------------------------------------------ |
        /// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \text{sgn}(\texttt{arg}[i_1,\dots,i_n])\f$ |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status             |
        /// | ------- | ------------------ |
        /// | NGVM    | Fully implemented. |
        class Sign : public UnaryElementwiseArithmetic
        {
        public:
            /// \brief Constructs an elementwise sign operation.
            ///
            /// \param arg Node that produces the input tensor.
            Sign(const std::shared_ptr<Node>& arg)
                : UnaryElementwiseArithmetic(arg)
            {
            }

            virtual std::string description() const override { return "Sign"; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;
        };
    }
}
