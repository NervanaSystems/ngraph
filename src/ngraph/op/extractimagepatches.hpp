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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            class NGRAPH_API ExtractImagePatches : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ExtractImagePatches", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ExtractImagePatches() = default;
                /// \brief Constructs a ExtractImagePatches operation
                ///
                /// \param data 4-D Input data to extract image patches
                /// \param sizes Patch size in the format of [size_rows, size_cols]
                /// \param strides Patch movement stride in the format of [stride_rows, stride_cols]
                /// \param rates Element seleciton rate for creating a patch. in the format of
                /// [rate_rows, rate_cols] \param padding Padding type. it can be any value from
                /// valid, same_lower, same_upper
                ExtractImagePatches(const Output<Node>& data,
                                    const Shape sizes,
                                    const Strides strides,
                                    const Shape rates,
                                    const PadType padding);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const Shape& get_sizes() const { return m_patch_sizes; }
                void set_sizes(const Shape _sizes) { m_patch_sizes = _sizes; }
                const Strides& get_strides() const { return m_patch_movement_strides; }
                void set_strides(const Strides _strides) { m_patch_movement_strides = _strides; }
                const Shape& get_rates() const { return m_patch_selection_rates; }
                void set_rates(const Shape _rates) { m_patch_selection_rates = _rates; }
                const PadType& get_padding() const { return m_padding; }
                void set_padding(PadType _padding) { m_padding = _padding; }
            private:
                Shape m_patch_sizes;
                Strides m_patch_movement_strides;
                Shape m_patch_selection_rates;
                PadType m_padding;
            };
        } // namespace v3
        using v3::ExtractImagePatches;
    } // namespace op
} // namespace ngraph
