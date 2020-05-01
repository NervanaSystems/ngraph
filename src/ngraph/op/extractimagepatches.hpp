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
            typedef struct
            {
                Shape patch_sizes;
                Strides patch_movement_strides;
                Shape patch_selection_rates;
                PadType padding;
            } ExtractImagePatchesAttrs;

            class NGRAPH_API ExtractImagePatches : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ExtractImagePatches", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ExtractImagePatches() = default;
                /// \brief Constructs a ExtractImagePatches operation
                ///
                /// \param data Input data to extract image patches
                /// \param attrs        ExtractImagePatches attributes
                ExtractImagePatches(const Output<Node>& data,
                                    const ExtractImagePatchesAttrs& attrs);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                const ExtractImagePatchesAttrs& get_attrs() const { return m_attrs; }
                void set_attrs(const Shape& sizes,
                               const Strides& strides,
                               const Shape& rates,
                               const std::string& str_padding)
                {
                    m_attrs.patch_sizes = sizes;
                    m_attrs.patch_movement_strides = strides;
                    m_attrs.patch_selection_rates = rates;
                    if (str_padding == "valid")
                    {
                        m_attrs.padding = PadType::VALID;
                    }
                    else if (str_padding == "same_lower")
                    {
                        m_attrs.padding = PadType::SAME_LOWER;
                    }
                    else if (str_padding == "same_upper")
                    {
                        m_attrs.padding = PadType::SAME_UPPER;
                    }
                    else
                    {
                        m_attrs.padding = PadType::NOTSET;
                    }
                }

                static ExtractImagePatchesAttrs
                    CreateExtractImagePatchesAttrs(const Shape& sizes,
                                                   const Strides& strides,
                                                   const Shape& rates,
                                                   const std::string& str_padding)
                {
                    ExtractImagePatchesAttrs attrs;
                    attrs.patch_sizes = sizes;
                    attrs.patch_movement_strides = strides;
                    attrs.patch_selection_rates = rates;
                    if (str_padding == "valid")
                    {
                        attrs.padding = PadType::VALID;
                    }
                    else if (str_padding == "same_lower")
                    {
                        attrs.padding = PadType::SAME_LOWER;
                    }
                    else if (str_padding == "same_upper")
                    {
                        attrs.padding = PadType::SAME_UPPER;
                    }
                    else
                    {
                        attrs.padding = PadType::NOTSET;
                    }
                    return attrs;
                }

            private:
                ExtractImagePatchesAttrs m_attrs;
            };
        } // namespace v3
        using v3::ExtractImagePatches;
        using v3::ExtractImagePatchesAttrs;
    } // namespace op
} // namespace ngraph