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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Operation which "adds" axes to an input tensor, replicating elements from the input as needed along the new axes.
        class Broadcast : public Op
        {
        public:
            /// \brief Constructs a broadcast operation.
            ///
            /// \param arg             Node that produces the input tensor to be broadcast.
            /// \param broadcast_shape Node that produces the shape of the output tensor.
            /// \param broadcast_axes  Node that produces the axis positions (0-based) in the
            ///                         broadcast_shape that are being broadcast. The remaining
            ///                         rest of the dimensions in the broadcast shape must match
            ///                         the dimensions of arg, in order.
            Broadcast(const std::shared_ptr<Node>& arg,
                      const std::shared_ptr<Node>& broadcast_shape,
                      const std::shared_ptr<Node>& broadcast_axes);

            /// \brief Constructs a conversion operation.
            ///
            /// \param arg             Node that produces the input tensor to be broadcast.
            /// \param broadcast_shape The shape of the output tensor.
            /// \param broadcast_axes  The axis positions (0-based) in the result that are being broadcast. The
            ///                         remaining axes in shape must be the same as the shape of arg.
            Broadcast(const std::shared_ptr<Node>& arg,
                      const Shape& broadcast_shape,
                      const AxisSet& broadcast_axes);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            Shape get_broadcast_shape() const;
            bool broadcast_shape_is_constant() const;

            /// \return A set containing the indices of the broadcast axes (0-based).
            AxisSet get_broadcast_axes() const;
            bool broadcast_axes_are_constant() const;
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
        };
    }
}
