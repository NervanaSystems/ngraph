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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

#pragma once

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            class NGRAPH_API BroadcastBase : public Op
            {
            protected:
                BroadcastBase() = default;
                /// \brief Constructs a broadcast operation.
                ///
                /// \param arg            The input tensor to be broadcast.
                /// \param target_shape   The shape of the output tensor.
                /// \param axes_mapping   The axis positions (0-based) in the result that correspond
                ///                       to input axes.
                /// \param broadcast_spec Broadcast specification to use for determining broadcast
                ///                       axes. 'axes_mapping' should not be provided if mode other
                ///
                BroadcastBase(const Output<Node>& arg,
                              const Output<Node>& target_shape,
                              const Output<Node>& axes_mapping,
                              const AutoBroadcastSpec& broadcast_spec = AutoBroadcastSpec());

                /// \brief Constructs a broadcast operation.
                ///
                /// \param arg            The input tensor to be broadcast.
                /// \param target_shape   The shape of the output tensor.
                /// \param broadcast_spec Broadcast specification to use for determining broadcast
                ///                       axes
                BroadcastBase(const Output<Node>& arg,
                              const Output<Node>& target_shape,
                              const AutoBroadcastSpec& broadcast_spec =
                                  AutoBroadcastSpec(AutoBroadcastType::NUMPY));

            public:
                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;
                /// \return Broadcast Specification.
                const AutoBroadcastSpec& get_broadcast_spec() const { return m_broadcast_spec; }
                void set_broadcast_spec(const AutoBroadcastSpec& broadcast_spec)
                {
                    m_broadcast_spec = broadcast_spec;
                }
                /// \return true and the AxisSet if broadcast axes can be fully determined.
                virtual std::pair<bool, AxisSet> get_broadcast_axes() const;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
                AutoBroadcastSpec m_broadcast_spec;
            };
        }
    }
}
