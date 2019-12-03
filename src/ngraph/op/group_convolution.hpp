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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Batched convolution operation, with optional window dilation and stride.
            class NGRAPH_API GroupConvolution : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"GroupConvolution", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched convolution operation.
                GroupConvolution() = default;
                /// \brief Constructs a batched convolution operation.
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param strides The strides.<br>
                /// `[f]`
                /// \param dilations The dilations.<br>
                /// `[f]`
                /// \param pads_begin The beginning of padding shape.<br>
                /// `[f]`
                /// \param pads_end The end of padding shape.<br>
                /// `[f]`
                /// \param auto_pad The pad type for automatically computing padding sizes.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                GroupConvolution(const Output<Node>& data_batch,
                                 const Output<Node>& filters,
                                 const Strides& strides,
                                 const CoordinateDiff& pads_begin,
                                 const CoordinateDiff& pads_end,
                                 const Strides& dilations,
                                 const PadType& auto_pad = PadType::EXPLICIT);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const NodeVector& deltas) override;

                /// \return The strides.
                const Strides& get_strides() const { return m_strides; }
                void set_strides(const Strides& strides) { m_strides = strides; }
                /// \return The dilations.
                const Strides& get_dilations() const { return m_dilations; }
                void set_dilations(const Strides& dilations) { m_dilations = dilations; }
                /// \return The padding-below sizes (possibly negative).
                const CoordinateDiff& get_pads_begin() const { return m_pads_begin; }
                void set_pads_begin(const CoordinateDiff& pads_begin) { m_pads_begin = pads_begin; }
                /// \return The padding-above sizes (possibly negative).
                const CoordinateDiff& get_pads_end() const { return m_pads_end; }
                void set_adding_above(const CoordinateDiff& pads_end) { m_pads_end = pads_end; }
                /// \return The pad type for convolution.
                const PadType& get_auto_pad() const { return m_auto_pad; }
                void set_auto_pad(const PadType& auto_pad) { m_auto_pad = auto_pad; }
                /// \return The default value for Convolution.
                virtual std::shared_ptr<Node> get_default_value() const override;

            protected:
                Strides m_strides;
                Strides m_dilations;
                CoordinateDiff m_pads_begin;
                CoordinateDiff m_pads_end;
                PadType m_auto_pad;
            };

            /// \brief Data batch backprop for batched convolution operation.
            class NGRAPH_API GroupConvolutionBackpropData : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"GroupConvolutionBackpropData", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched-convolution data batch-backprop operation.
                GroupConvolutionBackpropData() = default;
                // clang-format off
                /// \brief Constructs a batched-convolution data batch-backprop operation.
                ///
                /// \param data            The node producing data from forward-prop.
                /// \param filter          The node producing the filter from forward-prop.
                /// \param output_shape    The shape of the data batch from forward-prop.
                /// \param strides         The strides from forward-prop.
                /// \param pads_begin      The padding-below sizes from forward-prop.
                /// \param pads_end        The padding-above sizes from forward-prop.
                /// \param dilations       The dilations from forward-prop.
                /// \param auto_pad        The pad type for automatically computing padding sizes.
                /// \param output_padding  The output padding adds additional amount of paddings per each spatial axis in the output tensor.
                // clang-format on
                GroupConvolutionBackpropData(const Output<Node>& data,
                                             const Output<Node>& filter,
                                             const Output<Node>& output_shape,
                                             const Strides& strides,
                                             const CoordinateDiff& pads_begin,
                                             const CoordinateDiff& pads_end,
                                             const Strides& dilations,
                                             const PadType& auto_pad = PadType::EXPLICIT,
                                             const CoordinateDiff& output_padding = {});

                // clang-format off
                /// \brief Constructs a batched-convolution data batch-backprop operation.
                ///
                /// \param data            The node producing data from forward-prop.
                /// \param filter          The node producing the filter from forward-prop.
                /// \param strides         The strides from forward-prop.
                /// \param pads_begin      The padding-below sizes from forward-prop.
                /// \param pads_end        The padding-above sizes from forward-prop.
                /// \param dilations       The dilations from forward-prop.
                /// \param auto_pad        The pad type for automatically computing padding sizes.
                /// \param output_padding  The output padding adds additional amount of paddings per each spatial axis in the output tensor.
                // clang-format on
                GroupConvolutionBackpropData(const Output<Node>& data,
                                             const Output<Node>& filter,
                                             const Strides& strides,
                                             const CoordinateDiff& pads_begin,
                                             const CoordinateDiff& pads_end,
                                             const Strides& dilations,
                                             const PadType& auto_pad = PadType::EXPLICIT,
                                             const CoordinateDiff& output_padding = {});

                void validate_and_infer_types() override;

                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const NodeVector& deltas) override;
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                /// \return The data batch shape.
                const PartialShape get_output_shape() const;
                void set_output_shape(const Shape& output_shape);
                /// \return The strides from the forward prop.
                const Strides& get_strides() const { return m_strides; }
                void set_strides(const Strides& strides) { m_strides = strides; }
                /// \return The dilations from the forward prop.
                const Strides& get_dilations() const { return m_dilations; }
                void set_dilations(const Strides& dilations) { m_dilations = dilations; }
                /// \return The padding-below sizes (possibly negative) from the forward prop.
                const CoordinateDiff& get_pads_begin() const { return m_pads_begin; }
                void set_pads_begin(const CoordinateDiff& pads_begin) { m_pads_begin = pads_begin; }
                /// \return The padding-above sizes (possibly negative) from the forward prop.
                const CoordinateDiff& get_pads_end() const { return m_pads_end; }
                void set_pads_end(const CoordinateDiff& pads_end) { m_pads_end = pads_end; }
                /// \return The auto pad.
                const PadType& get_auto_pad() const { return m_auto_pad; }
                void set_auto_pad(const PadType& auto_pad) { m_auto_pad = auto_pad; }
                /// \return The output padding.
                const CoordinateDiff& get_output_padding() const { return m_output_padding; }
                void set_output_padding(const CoordinateDiff& output_padding)
                {
                    m_output_padding = output_padding;
                }

            protected:
                Strides m_strides;
                Strides m_dilations;
                CoordinateDiff m_pads_begin;
                CoordinateDiff m_pads_end;
                PadType m_auto_pad;
                CoordinateDiff m_output_padding;
            };
        } // namespace v1

    } // namespace op
} // namespace ngraph
