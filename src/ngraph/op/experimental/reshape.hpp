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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Tensor dynamic reshape operation.
            ///
            /// "Converts" an input tensor into a new shape with the same number of elements.
            /// This op does not touch the actual data. If needed, use Transpose for that purpose.
            ///
            class Reshape : public Op
            {
            public:
            public:
                NGRAPH_API
                static constexpr NodeTypeInfo type_info{"DynReshape", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Reshape() = default;
                /// \brief Constructs a dynamic reshape operation. This operation does not perform
                ///        transpose.
                ///
                /// \param arg The tensor to be reshaped.
                /// \param pattern The node that defines output shape pattern.
                ///        If the input shape is \f$(a_0,\dots,a_{k-1})\f$ then the output shape
                ///        must
                ///        be of the form \f$(b_0,\dots,b_{j-1})\f$ where \f$\Pi(a_i) = \Pi(b_i)\f$.
                ///        A value of -1 is allowed for at most one dimension, in which case the
                ///        dimension size is inferred based on element count of input tensor.
                Reshape(const Output<Node>& arg, const Output<Node>& pattern);

                void validate_and_infer_types() override;

                size_t get_version() const override { return 1; }
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const NodeVector& deltas) override;
            };
        }
    }
}