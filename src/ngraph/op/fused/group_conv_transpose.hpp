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

#include <cstdlib>
#include <memory>

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Group Transpose Convolution (Deconvolution)
            class NGRAPH_API GroupConvolutionTranspose : public util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GroupConvolutionTranspose", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GroupConvolutionTranspose() = default;
                ///
                /// \brief      Constructs GroupConvolutionTranspose operation.
                ///
                /// \param[in]  data            The node producing input data.
                /// \param[in]  filters         The node producing filters data.
                /// \param[in]  strides         The strides along each feature axis.
                /// \param[in]  dilations       The dilations along each feature axis.
                /// \param[in]  padding_begin   The padding added at the beggining of each feature
                /// axis.
                /// \param[in]  padding_end     The padding added at the end of each feature axis.
                /// \param[in]  output_padding  The zero-padding (adjustment) added to one side of
                /// the
                ///                             output.
                /// \param[in]  groups          The number of groups the input channels and output
                ///                             channels are divided into.
                /// \param[in]  pad_type        The provided padding type.
                /// \param[in]  output_shape    The output shape. When provided padding values are
                ///                             automatically inferred.
                ///
                GroupConvolutionTranspose(const Output<Node>& data,
                                          const Output<Node>& filters,
                                          const Strides& strides,
                                          const Strides& dilations,
                                          const CoordinateDiff& padding_begin,
                                          const CoordinateDiff& padding_end,
                                          const CoordinateDiff& output_padding,
                                          const std::size_t groups = 1UL,
                                          const PadType& pad_type = PadType::EXPLICIT,
                                          const Shape& output_shape = Shape{});

                ///
                /// \brief      Constructs GroupConvolutionTranspose operation.
                ///
                /// \param[in]  data      The node producing input data.
                /// \param[in]  filters   The node producing filters data.
                /// \param[in]  groups    The number of groups the input channels and output
                /// channels
                ///                       are divided into.
                ///
                GroupConvolutionTranspose(const Output<Node>& data,
                                          const Output<Node>& filters,
                                          const std::size_t groups = 1UL);

                ///
                /// \brief      Constructs GroupConvolutionTranspose operation.
                ///
                /// \param[in]  data            The node producing input data.
                /// \param[in]  filters         The node producing filters data.
                /// \param[in]  strides         The strides along each feature axis.
                /// \param[in]  dilations       The dilations along each feature axis.
                /// \param[in]  output_padding  The zero-padding (adjustment) added to one side of
                /// the
                ///                             output.
                /// \param[in]  output_shape    The output shape. When provided padding values are
                ///                             automatically inferred.
                /// \param[in]  groups          The number of groups the input channels and output
                ///                             channels are divided into.
                ///
                GroupConvolutionTranspose(const Output<Node>& data,
                                          const Output<Node>& filters,
                                          const Strides& strides,
                                          const Strides& dilations,
                                          const CoordinateDiff& output_padding,
                                          const Shape& output_shape,
                                          const std::size_t groups = 1UL);

                ///
                /// \brief      Constructs GroupConvolutionTranspose operation.
                ///
                /// \param[in]  data            The node producing input data.
                /// \param[in]  filters         The node producing filters data.
                /// \param[in]  output_shape    The output shape. When provided padding values are
                ///                             automatically inferred.
                /// \param[in]  groups          The number of groups the input channels and output
                ///                             channels are divided into.
                ///
                GroupConvolutionTranspose(const Output<Node>& data,
                                          const Output<Node>& filters,
                                          const Shape& output_shape,
                                          const std::size_t groups = 1UL);

                Output<Node> get_data() { return input_value(0); }
                Output<Node> get_filters() { return input_value(1); }
                const Strides& get_strides() const { return m_strides; }
                const Strides& get_dilations() const { return m_dilations; }
                const CoordinateDiff& get_padding_begin() const { return m_padding_begin; }
                const CoordinateDiff& get_padding_end() const { return m_padding_end; }
                const CoordinateDiff& get_output_padding() const { return m_output_padding; }
                std::size_t get_groups() const { return m_groups; }
                const PadType& get_pad_type() const { return m_pad_type; }
                const Shape& get_output_shape() const { return m_output_shape; }
                virtual void pre_validate_and_infer_types() override;
                virtual void post_validate_and_infer_types() override;
                virtual NodeVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            private:
                ///
                /// \brief      Calculate the shape of the data batch from forward propagation.
                ///
                /// \return     The data batch shape.
                ///
                Shape get_data_batch_shape() const;

                Strides m_strides;
                Strides m_dilations;
                CoordinateDiff m_padding_begin;
                CoordinateDiff m_padding_end;
                CoordinateDiff m_output_padding;
                std::size_t m_groups;
                PadType m_pad_type;
                Shape m_output_shape;
            };
        }
        using v0::GroupConvolutionTranspose;
    }
}
