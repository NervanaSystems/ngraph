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
        /// \brief Elementwise multiplication operation.
        class Multiply : public util::BinaryElementwiseArithmetic
        {
        public:
            /// \brief Constructs a multiplication operation.
            ///
            /// \param arg0 Output that produces the first input tensor.
            /// \param arg1 Output that produces the second input tensor.
            Multiply(const NodeOutput& arg0, const NodeOutput& arg1);

            virtual std::shared_ptr<Node>
                copy_with_new_source_outputs(const OutputVector& new_source_outputs) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const OutputVector& deltas) override;
            virtual bool is_commutative() override { return true; }
        };
    };

    std::shared_ptr<ngraph::Node> operator*(const NodeOutput& arg0, const NodeOutput& arg1);
}
