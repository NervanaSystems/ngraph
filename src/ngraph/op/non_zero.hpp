//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
        namespace v3
        {
            /// \brief NonZero operation returning indices of non-zero elements in the input tensor.
            ///
            /// \note The indices are returned by-dimension in row-major order. For example
            ///       the following output contains 3 indices of a 3D input tensor elements:
            ///       [[0, 0, 2],
            ///        [0, 1, 1],
            ///        [0, 1, 2]]
            ///       The values point to input elements at [0,0,0], [0,1,1] and [2,1,2]
            class NGRAPH_API NonZero : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"NonZero", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a NonZero operation.
                NonZero() = default;
                /// \brief Constructs a NonZero operation.
                ///
                /// \param arg Node that produces the input tensor.
                NonZero(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v3::NonZero;
    } // namespace op
} // namespace ngraph
