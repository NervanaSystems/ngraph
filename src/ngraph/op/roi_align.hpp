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
            class NGRAPH_API ROIAlign : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ROIAlign", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ROIAlign() = default;
                /// \brief Constructs a ROIAlign node matching the ONNX ROIAlign specification
                ///
                /// \param input           Input feature map {N, C, H, W}
                /// \param rois            Regions of interest to pool over
                /// \param batch_indices   Indices of images in the batch matching
                ///                        the number or ROIs
                /// \param pooled_h        Height of the ROI output features
                /// \param pooled_w        Width of the ROI output features
                /// \param sampling_ratio  Number of sampling points used to compute
                ///                        an output element
                /// \param spatial_scale   Spatial scale factor used to translate ROI coordinates
                /// \param mode            Method of pooling - 'avg' or 'max'
                ROIAlign(const Output<Node>& input,
                         const Output<Node>& rois,
                         const Output<Node>& batch_indices,
                         const size_t pooled_h,
                         const size_t pooled_w,
                         const size_t sampling_ratio,
                         const float spatial_scale,
                         const std::string& mode);

                virtual void validate_and_infer_types() override;
                virtual bool visit_attributes(AttributeVisitor& visitor) override;
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                size_t get_pooled_h() const { return m_pooled_h; }
                size_t get_pooled_w() const { return m_pooled_w; }
                size_t get_sampling_ratio() const { return m_sampling_ratio; }
                float get_spatial_scale() const { return m_spatial_scale; }
                const std::string& get_mode() const { return m_mode; }
            private:
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
