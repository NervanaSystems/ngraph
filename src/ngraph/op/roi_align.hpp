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

#include <string>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            /// \brief ROIAlign operation.
            class NGRAPH_API ROIAlign : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ROIAlign", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a ROIAlign operation.
                ROIAlign() = default;

                /// \brief Constructs a ROIAlign operation.
                ///
                /// \param arg data that produces the input feature map tensor.
                /// \param arg rois that produces the input ROIs coordinates tensor.
                /// \param arg batch_indices that produces the input batch indices tensor.
                ROIAlign(const Output<Node>& data,
                         const Output<Node>& rois,
                         const Output<Node>& batch_indices,
                         size_t pooled_h,
                         size_t pooled_w,
                         size_t sampling_ratio,
                         float spatial_scale,
                         const std::string& mode);

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

            protected:
                size_t m_pooled_h;
                size_t m_pooled_w;
                size_t m_sampling_ratio;
                float m_spatial_scale;
                std::string m_mode;
            };
        }
        using v3::ROIAlign;
    }
}
