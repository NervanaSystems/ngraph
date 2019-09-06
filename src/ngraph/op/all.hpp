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

#include "ngraph/op/util/logical_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Logical "all" reduction operation.
        class All : public util::LogicalReduction
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs an "all" reduction operation.
            All() = default;
            /// \brief Constructs an "all" reduction operation.
            ///
            /// \param arg The tensor to be reduced.
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            All(const Output<Node>& arg, const AxisSet& reduction_axes);
            /// \brief Constructs an "all" reduction operation.
            ///
            /// \param arg The tensor to be reduced.
            /// \param reduction_axes The axis positions (0-based) to be eliminated.
            All(const Output<Node>& arg, const Output<Node>& reduction_axes);

            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The default value for All.
            virtual std::shared_ptr<Node> get_default_value() const override;
        };
    }
}
