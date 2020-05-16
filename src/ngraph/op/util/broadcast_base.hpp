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
#include "ngraph/axis_vector.hpp"
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
                /// \param broadcast_mode Broadcast specification to use for determining broadcast
                ///                       axes. 'axes_mapping' should not be provided if mode other
                ///
                BroadcastBase(const Output<Node>& arg,
                              const Output<Node>& target_shape,
                              const Output<Node>& axes_mapping,
                              const BroadcastModeSpec& broadcast_mode = BroadcastType::EXPLICIT);

                /// \brief Constructs a broadcast operation.
                ///
                /// \param arg            The input tensor to be broadcast.
                /// \param target_shape   The shape of the output tensor.
                /// \param broadcast_mode Broadcast specification to use for determining broadcast
                ///                       axes
                BroadcastBase(const Output<Node>& arg,
                              const Output<Node>& target_shape,
                              const BroadcastModeSpec& broadcast_mode = BroadcastType::NUMPY);

            public:
                void validate_and_infer_types() override;
                /// \return true and the AxisSet if broadcast axes can be fully determined.
                virtual std::pair<bool, AxisSet> get_broadcast_axes() const;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
                BroadcastModeSpec m_mode;

                bool evaluate_broadcast(const HostTensorPtr& arg0,
                                        const HostTensorPtr& out,
                                        const std::pair<bool, AxisSet> pair_broadcast_axes,
                                        const Shape output_shape);

                template <element::Type_t ET>
                bool evaluate(const HostTensorPtr& arg0,
                              const HostTensorPtr& out,
                              const AxisSet& broadcast_axes);

                PartialShape
                    get_result_shape_numpy_pdpd(const Shape& arg0_shape,
                                                const Shape& target_shape,
                                                const op::BroadcastModeSpec& broadcast_spec);
                static std::pair<bool, AxisSet>
                    get_broadcast_axes_numpy_pdpd(const Shape& arg_shape,
                                                  const Shape& result_shape,
                                                  const op::BroadcastModeSpec& broadcast_spec);

                static std::pair<bool, AxisSet>
                    get_broadcast_axes_none(const AxisVector axes_mapping_val,
                                            const size_t target_shape);

                void validate_target_shape_none(const Shape& arg_shape,
                                                const AxisVector& axes_mapping_val,
                                                const Shape& target_shape);

                Shape get_target_shape(const HostTensorPtr& input1);
            };
        }
    }
}
