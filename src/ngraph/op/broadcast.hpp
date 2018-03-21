/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Operation which "adds" axes to an input tensor, replicating elements from the input as needed along the new axes.
        class Broadcast : public util::RequiresTensorViewArgs
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

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override
            {
                if (new_args.size() != 1)
                {
                    throw ngraph_error("Incorrect number of new arguments");
                }
                return std::make_shared<Broadcast>(new_args.at(0), m_shape, m_broadcast_axes);
            }

            /// \return A set containing the indices of the broadcast axes (0-based).
            const AxisSet& get_broadcast_axes() const { return m_broadcast_axes; }
            const Shape& get_broadcast_shape() const { return m_shape; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;

            Shape m_shape;
            AxisSet m_broadcast_axes;
        };
    }
}
