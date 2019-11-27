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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            class DeformablePSROIPooling : public Op
            {
            public:
                NGRAPH_API
                static constexpr NodeTypeInfo type_info{"DeformablePSROIPooling", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                DeformablePSROIPooling() = default;
                /// \brief Constructs a PSROIPooling operation
                ///
                /// \param input          Input feature map {N, C, ...}
                /// \param coords         Coordinates of bounding boxes
                /// \param output_dim     Output channel number
                /// \param group_size     Number of groups to encode position-sensitive scores
                /// \param spatial_scale  Ratio of input feature map over input image size
                /// \param spatial_bins_x Numbers of bins to divide the input feature maps over
                /// width
                /// \param spatial_bins_y Numbers of bins to divide the input feature maps over
                /// height
                /// \param mode           Mode of pooling - Avg or Bilinear
                // TODO
                DeformablePSROIPooling(const Output<Node>& input,
                                       const Output<Node>& coords,
                                       const Output<Node>& offsets,
                                       const int64_t output_dim,
                                       const int64_t group_size,
                                       const float spatial_scale,
                                       const std::string mode = "bilinear_deformable",
                                       int64_t spatial_bins_x = 1,
                                       int64_t spatial_bins_y = 1,
                                       bool no_trans = true,
                                       float trans_std = 1,
                                       int64_t part_size = 1);

                DeformablePSROIPooling(const Output<Node>& input,
                                       const Output<Node>& coords,
                                       const int64_t output_dim,
                                       const int64_t group_size,
                                       const float spatial_scale,
                                       const std::string mode = "bilinear_deformable",
                                       int64_t spatial_bins_x = 1,
                                       int64_t spatial_bins_y = 1,
                                       bool no_trans = true,
                                       float trans_std = 1,
                                       int64_t part_size = 1);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                int64_t get_output_dim() const { return m_output_dim; }
                int64_t get_group_size() const { return m_group_size; }
                float get_spatial_scale() const { return m_spatial_scale; }
                const std::string& get_mode() const { return m_mode; }
                int64_t get_spatial_bins_x() const { return m_spatial_bins_x; }
                int64_t get_spatial_bins_y() const { return m_spatial_bins_y; }
                bool get_no_trans() const { return m_no_trans; }
                float get_trans_std() const { return m_trans_std; }
                int64_t get_part_size() const { return m_part_size; }
            private:
                int64_t m_output_dim;
                int64_t m_group_size;
                float m_spatial_scale;
                std::string m_mode;
                int64_t m_spatial_bins_x;
                int64_t m_spatial_bins_y;
                bool m_no_trans;
                float m_trans_std;
                int64_t m_part_size;
            };
        }
    }
}
