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
        class PriorBoxClustered : public Op
        {
        public:
            /// \brief Constructs a PriorBoxClustered operation
            ///
            /// \param layer_shape    Shape of layer for which prior boxes are computed
            /// \param image_shape    Shape of image to which prior boxes are scaled
            /// \param num_priors     Number of prior boxes
            /// \param widths         Desired widths of prior boxes
            /// \param heights        Desired heights of prior boxes
            /// \param clip           Clip output to [0,1]
            /// \param step_widths    Distance between prior box centers
            /// \param step_heights   Distance between prior box centers
            /// \param offset         Box offset relative to top center of image
            /// \param variances      Values to adjust prior boxes with
            PriorBoxClustered(const std::shared_ptr<Node>& layer_shape,
                              const std::shared_ptr<Node>& image_shape,
                              const size_t num_priors,
                              const std::vector<float>& widths,
                              const std::vector<float>& heights,
                              const bool clip,
                              const float step_widths,
                              const float step_heights,
                              const float offset,
                              const std::vector<float>& variances);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            size_t m_num_priors;
            std::vector<float> m_widths;
            std::vector<float> m_heights;
            bool m_clip;
            float m_step_widths;
            float m_step_heights;
            float m_offset;
            std::vector<float> m_variances;
        };
    }
}
