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
        /// \brief Elementwise division operation.
        class Divide : public util::BinaryElementwiseArithmetic
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a division operation.
            Divide() = default;
            /// \brief Constructs a division operation.
            ///
            /// \param arg0 Node that produces the first input tensor.
            /// \param arg1 Node that produces the second input tensor.
            /// \param pythondiv Use Python style rounding for integral type
            /// \param autob Auto broadcast specification
            Divide(const Output<Node>& arg0,
                   const Output<Node>& arg1,
                   bool pythondiv,
                   const AutoBroadcastSpec& autob = AutoBroadcastSpec());

            /// \brief Constructs a division operation.
            ///
            /// \param arg0 Node that produces the first input tensor.
            /// \param arg1 Node that produces the second input tensor.
            /// \param autob Auto broadcast specification
            Divide(const Output<Node>& arg0,
                   const Output<Node>& arg1,
                   const AutoBroadcastSpec& autob = AutoBroadcastSpec());

            bool is_pythondiv() const { return m_pythondiv; }
            void set_is_pythondiv(bool pythondiv) { m_pythondiv = pythondiv; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        protected:
            bool m_pythondiv{true};
        };
    }

    std::shared_ptr<Node> operator/(const Output<Node>& arg0, const Output<Node>& arg1);
}
