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
        class ROIPooling : public Op
        {
        public:
            /// \brief Constructs a ROIPooling operation
            ///
            /// \param input          Input feature map {N, C, ...}
            /// \param coords         Coordinates of bounding boxes
            /// \param output_size    Height/Width of ROI output features
            /// \param spatial_scale  Ratio of input feature map over input image size
            /// \param kind           Kind of pooling - Max or Bilinear
            ROIPooling(const std::shared_ptr<Node>& input,
                       const std::shared_ptr<Node>& coords,
                       const Shape& output_size,
                       const float spatial_scale,
                       const std::string& kind);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            Shape m_output_size;
            float m_spatial_scale;
            std::string m_kind;
        };
    }
}
