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
            /// \brief Constructs a conversion operation.
            ///
            /// \param arg            Node that produces the input tensor to be broadcast.
            /// \param shape          The shape of the output tensor.
            /// \param broadcast_axes The axis positions (0-based) in the result that are being broadcast. The
            ///                        remaining axes in shape must be the same as the shape of arg.
            Broadcast(const std::shared_ptr<Node>& arg,
                      const Shape& shape,
                      const AxisSet& broadcast_axes);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return A set containing the indices of the broadcast axes (0-based).
            const AxisSet& get_broadcast_axes() const { return m_broadcast_axes; }
            const Shape& get_broadcast_shape() const { return m_shape; }
        protected:
            Broadcast(const std::string& node_type,
                      const NodeVector& args,
                      const Shape& shape,
                      const AxisSet& broadcast_axes);

            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

            virtual void infer_shape() {}
            Shape m_shape;
            AxisSet m_broadcast_axes;
        };

        /// \brief Broadcast arg to the same shape as like_arg.
        class BroadcastLike : public Broadcast
        {
        public:
            /// \brief Broadcast arg to the same shape as like_arg.
            ///
            /// Once the shape of like_arg is known, this op will be replaced with an equivalent
            /// Broadcast op.
            ///
            /// \param arg The argument to be broadcast.
            /// \param like_arg Provides the shape for the result.
            /// \param broadcast_axes indicates which axes will be broadcast. If empty,
            /// arg must be scalar and all axes are broadcast.
            BroadcastLike(const std::shared_ptr<Node>& arg,
                          const std::shared_ptr<Node>& like_arg,
                          const AxisSet& broadcast_axes);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            void infer_shape() override;

        protected:
            AxisSet m_initial_broadcast_axes;
        };
    }
}
