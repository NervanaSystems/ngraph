//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            // clang-format off
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
            // clang-format on
            class NGRAPH_API Power : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Power", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Power()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NONE)
                {
                }
                /// \brief Constructs an exponentiation operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Power(const Output<Node>& arg0,
                      const Output<Node>& arg1,
                      const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };
        } // namespace v0

        namespace v1
        {
            // clang-format off
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
            // clang-format on
            class NGRAPH_API Power : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Power", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Power()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NUMPY)
                {
                }

                /// \brief Constructs an exponentiation operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Power(const Output<Node>& arg0,
                      const Output<Node>& arg1,
                      const AutoBroadcastSpec& auto_broadcast =
                          AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
                size_t get_version() const override { return 1; }
            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };
        } // namespace v1

        using v0::Power;
    }
}
