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

#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Min-reduction operation.
            class NGRAPH_API Min : public util::ArithmeticReduction
            {
            public:
                static constexpr NodeTypeInfo type_info{"Min", 0};

                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a "min" reduction operation.
                Min() = default;

                /// \brief Constructs a min-reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                Min(const Output<Node>& arg, const AxisSet& reduction_axes);

                /// \brief Constructs a "min" reduction operation.
                ///
                /// \param arg The tensor to be reduced.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                Min(const Output<Node>& arg, const Output<Node>& reduction_axes);

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                /// \return The default value for Min.
                virtual std::shared_ptr<Node> get_default_value() const override;
            };
        }

        namespace v1
        {
            class NGRAPH_API ReduceMin : public util::ArithmeticReductionKeepDims
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReduceMin", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a summation operation.
                ReduceMin() = default;
                /// \brief Constructs a summation operation.
                ///
                /// \param arg The tensor to be summed.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to 1 it holds axes that are used for reduction.
                ReduceMin(const Output<Node>& arg,
                          const Output<Node>& reduction_axes,
                          bool keep_dims = false);

                size_t get_version() const override { return 1; }
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
            };
        }

        using v0::Min;
    }
}
