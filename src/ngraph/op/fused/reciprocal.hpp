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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Reciprocal operation
        /// f(x) = 1 / x
        class NGRAPH_API Reciprocal : public ngraph::op::util::FusedOp
        {
        public:
            static constexpr NodeTypeInfo type_info{"Reciprocal", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            Reciprocal() = default;
            /// \brief Constructs a Reciprocal operation.
            ///
            /// \param data Input tensor
            Reciprocal(const Output<Node>& data);

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    } // namespace op
} // namespace ngraph
