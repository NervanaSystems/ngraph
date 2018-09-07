//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Abstract base class for arithmetic reduction operations, i.e., operations where chosen axes of the input tensors
            ///        are eliminated (reduced out) by repeated application of a particular binary arithmetic operation.
            class ArithmeticReduction : public Op
            {
            public:
                /// \brief Constructs an arithmetic reduction operation.
                ///
                /// \param arg Node that produces the first input tensor.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                ArithmeticReduction(const std::string& node_type,
                                    const std::shared_ptr<Node>& arg,
                                    const AxisSet& reduction_axes);

                void validate_and_infer_types() override;

                /// \return The axis positions (0-based) to be eliminated through reduction.
                const AxisSet& get_reduction_axes() const { return m_reduction_axes; }
            protected:
                AxisSet m_reduction_axes;
            };
        }
    }
}
