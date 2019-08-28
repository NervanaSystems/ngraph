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
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Operator performing Mean Variance Normalization
        ///
        class MVN : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            MVN() = default;
            /// \brief Constructs an MVN operation.
            ///
            /// \param data Input tensor with data
            /// \param normalize_variance flag that denotes whether to perform variance
            ///                           normalization.
            /// \param across_channels flag that denotes if mean values are shared across channels.
            /// \param eps the number to be added to the variance to avoid division by zero when
            ///            normalizing the value
            ///
            MVN(const Output<Node>& data,
                bool across_channels = true,
                bool normalize_variance = true,
                double eps = 1e-9);

            /// \brief Constructs an MVN operation.
            ///
            /// \param data Input tensor with data
            /// \param reduction_axes A list of axes, along which to reduce.
            /// \param normalize_variance flag that denotes whether to perform variance
            ///                           normalization.
            /// \param eps the number to be added to the variance to avoid division by zero when
            ///            normalizing the value
            ///
            MVN(const Output<Node>& data,
                AxisSet reduction_axes,
                bool normalize_variance = true,
                double eps = 1e-9);

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            double get_eps() const { return m_eps; }
            bool get_normalize_variance() const { return m_normalize_variance; }
            AxisSet get_reduction_axes() const { return m_reduction_axes; }
        private:
            const double m_eps;
            const bool m_across_channels;
            const bool m_normalize_variance;
            AxisSet m_reduction_axes;
        };
    } // namespace op
} // namespace ngraph
