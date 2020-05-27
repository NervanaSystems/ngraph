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
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief SpaceToBatch permutes data tensor blocks of spatial data into batch
            /// dimension.
            ///
            /// \note  Values from spatial blocks dimensions are moved in the batch dimension.
            ///
            ///        Output node produces a tensor with shape: tensor with shape
            ///        `[batch * block_shape[0] * block_shape[1] * ... * block_shape[N - 1],
            ///         (pads_begin[1] + D_1 + pads_end[1]) / block_shape[1],
            ///         (pads_begin[2] + D_2 + pads_end[2]) / block_shape[2], ...,
            ///         (pads_begin[N - 1] + D_{N - 1} + pads_end[N - 1]) / block_shape[N - 1]`
            ///         of the same type as `data` input.
            class NGRAPH_API SpaceToBatch : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"SpaceToBatch", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                SpaceToBatch() = default;

                /// \brief Constructs a SpaceToBatch operation.
                ///
                /// \param data Node producing the data tensor
                /// \param block_shape The sizes of the block of values to be moved
                /// \param pads_begin Specifies the padding for the beginning along each axis of
                /// `data` input
                /// \param pads_end Specifies the padding for the ending along each axis of `data`
                /// input.
                SpaceToBatch(const Output<Node>& data,
                             const Output<Node>& block_shape,
                             const Output<ngraph::Node>& pads_begin,
                             const Output<ngraph::Node>& pads_end);

                NodeVector decompose_op() const override;
                void pre_validate_and_infer_types() override;
                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                bool visit_attributes(AttributeVisitor& visitor) override;
            };
        }
        using v1::SpaceToBatch;
    }
}
