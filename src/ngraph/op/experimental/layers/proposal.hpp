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
        class Proposal : public Op
        {
        public:
            /// \brief Constructs a Proposal operation
            ///
            /// \param class_probs     Class probability scores
            /// \param class_logits    Class prediction logits
            /// \param image_shape     Shape of image
            /// \param base_size       Anchor sizes
            /// \param pre_nms_topn    Number of boxes before nms
            /// \param post_nms_topn   Number of boxes after nms
            /// \param nms_threshold   Threshold for nms
            /// \param feature_stride  Feature stride
            /// \param min_size        Minimum box size
            /// \param anchor_ratios   Ratios for anchor generation
            /// \param anchor_scales   Scales for anchor generation
            /// \param clip_before_nms Clip before NMs
            /// \param clip_after_nms  Clip after NMs
            /// \param normalize       Normalize boxes to [0,1]
            /// \param box_size_scale  Scale factor for scaling box size logits
            /// \param box_coord_scale Scale factor for scaling box coordiate logits
            /// \param algo            Calculation algorithm to use
            Proposal(const std::shared_ptr<Node>& class_probs,
                     const std::shared_ptr<Node>& class_logits,
                     const std::shared_ptr<Node>& image_shape,
                     const size_t base_size,
                     const size_t pre_nms_topn,
                     const size_t post_nms_topn,
                     const float nms_threshold,
                     const size_t feature_stride,
                     const size_t min_size,
                     const std::vector<float>& anchor_ratios,
                     const std::vector<float>& anchor_scales,
                     const bool clip_before_nms,
                     const bool clip_after_nms,
                     const bool normalize,
                     const float box_size_scale,
                     const float box_coord_scale,
                     const std::string& algo);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            size_t m_base_size;
            size_t m_pre_nms_topn;
            size_t m_post_nms_topn;
            float m_nms_threshold;
            size_t m_feature_stride;
            size_t m_min_size;
            std::vector<float> m_anchor_ratios;
            std::vector<float> m_anchor_scales;
            bool m_clip_before_nms;
            bool m_clip_after_nms;
            bool m_normalize;
            float m_box_size_scale;
            float m_box_coord_scale;
            std::string m_algo;
        };
    }
}
