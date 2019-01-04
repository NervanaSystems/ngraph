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

namespace ngraph
{
    namespace op
    {
        /// \brief Min-reduction operation.
        class Min : public util::ArithmeticReduction
        {
        public:
            /// \brief Constructs a min-reduction operation.
            ///
            /// \param arg The tensor view to be reduced.
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            Min(const std::shared_ptr<Node>& arg, const AxisSet& reduction_axes);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
