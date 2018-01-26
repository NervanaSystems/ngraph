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
        /// \brief Elementwise exponentiation operation.
        ///
        /// ## Inputs
        ///
        /// |        | Type                              | Description                                            |
        /// | ------ | --------------------------------- | ------------------------------------------------------ |
        /// | `arg0` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type.        |
        /// | `arg1` | \f$N[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of the same shape and element type as `arg0`. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                                                    |
        /// | ---------------------- | -------------------------------------------------------------------------------------------------------------- |
        /// | \f$N[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \texttt{arg0}[i_1,\dots,i_n]^{\texttt{arg1}[i_1,\dots,i_n]}\f$ |
        class Power : public BinaryElementwiseArithmetic
        {
        public:
            /// \brief Constructs an exponentiation operation.
            ///
            /// \param arg0 Node that produces the first input tensor.
            /// \param arg1 Node that produces the second input tensor.
            Power(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
                : BinaryElementwiseArithmetic("Power", arg0, arg1)
            {
            }

            virtual std::shared_ptr<Node> copy_with_new_args(const Nodes& new_args) const override
            {
                if (new_args.size() != 2)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<Power>(new_args.at(0), new_args.at(1));
            }
            bool is_functionally_identical(const Node&) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;
        };
    }
}
