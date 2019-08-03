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
        class PSROIPooling : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a PSROIPooling operation
            ///
            /// \param input          Input feature map {N, C, ...}
            /// \param coords         Coordinates of bounding boxes
            /// \param output_dim     Output channel number
            /// \param group_size     Number of groups to encode position-sensitive scores
            /// \param spatial_scale  Ratio of input feature map over input image size
            /// \param num_bins       Number of bins to divide the input feature maps
            /// \param kind           Kind of pooling - Avg or Bilinear
            PSROIPooling(const std::shared_ptr<Node>& input,
                         const std::shared_ptr<Node>& coords,
                         const size_t output_dim,
                         const size_t group_size,
                         const float spatial_scale,
                         const Shape& num_bins,
                         const std::string& kind);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_output_dim() const { return m_output_dim; }
            size_t get_group_size() const { return m_group_size; }
            float get_spatial_scale() const { return m_spatial_scale; }
            const Shape& get_num_bins() const { return m_num_bins; }
            const std::string& get_kind() const { return m_kind; }
        private:
            size_t m_output_dim;
            size_t m_group_size;
            float m_spatial_scale;
            Shape m_num_bins;
            std::string m_kind;
        };
    }
}
