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
        /// \brief Layer which generates prior boxes of specified sizes
        /// normalized to input image size
        class PriorBox : public Op
        {
        public:
            /// \brief Constructs a PriorBox operation
            ///
            /// \param layer_shape    Shape of layer for which prior boxes are computed
            /// \param image_shape    Shape of image to which prior boxes are scaled
            /// \param min_sizes      Desired min_sizess of prior boxes
            /// \param max_sizes      Desired max_sizess of prior boxes
            /// \param aspect_ratios  Aspect ratios of prior boxes
            /// \param clip           Clip output to [0,1]
            /// \param flip           Flip aspect ratios
            /// \param step           Distance between prior box centers
            /// \param offset         Box offset relative to top center of image
            /// \param variances      Values to adjust prior boxes with
            /// \param scale_all      Scale all sizes
            PriorBox(const std::shared_ptr<Node>& layer_shape,
                     const std::shared_ptr<Node>& image_shape,
                     const std::vector<float>& min_sizes,
                     const std::vector<float>& max_sizes,
                     const std::vector<float>& aspect_ratios,
                     const bool clip,
                     const bool flip,
                     const float step,
                     const float offset,
                     const std::vector<float>& variances,
                     const bool scale_all);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            std::vector<float> m_min_sizes;
            std::vector<float> m_max_sizes;
            std::vector<float> m_aspect_ratios;
            bool m_clip;
            bool m_flip;
            float m_step;
            float m_offset;
            std::vector<float> m_variances;
            bool m_scale_all;
        };
    }
}
