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
        // base_size       Anchor sizes
        // pre_nms_topn    Number of boxes before nms
        // post_nms_topn   Number of boxes after nms
        // nms_thresh      Threshold for nms
        // feat_stride     Feature stride
        // min_size        Minimum box size
        // ratio   Ratios for anchor generation
        // scale   Scales for anchor generation
        // clip_before_nms Clip before NMs
        // clip_after_nms  Clip after NMs
        // normalize       Normalize boxes to [0,1]
        // box_size_scale  Scale factor for scaling box size logits
        // box_coordinate_scale Scale factor for scaling box coordiate logits
        // framework            Calculation frameworkrithm to use
        struct ProposalAttrs
        {
            size_t base_size;
            size_t pre_nms_topn;
            size_t post_nms_topn;
            float nms_thresh = 0.0f;
            size_t feat_stride = 1;
            size_t min_size = 1;
            std::vector<float> ratio;
            std::vector<float> scale;
            bool clip_before_nms = false;
            bool clip_after_nms = false;
            bool normalize = false;
            float box_size_scale = 1.0f;
            float box_coordinate_scale = 1.0f;
            std::string framework;
        };

        class Proposal : public Op
        {
        public:
            /// \brief Constructs a Proposal operation
            ///
            /// \param class_probs     Class probability scores
            /// \param class_logits    Class prediction logits
            /// \param image_shape     Shape of image
            /// \param attrs           Proposal op attributes
            Proposal(const std::shared_ptr<Node>& class_probs,
                     const std::shared_ptr<Node>& class_logits,
                     const std::shared_ptr<Node>& image_shape,
                     const ProposalAttrs& attrs);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const ProposalAttrs& get_attrs() const { return m_attrs; }
        private:
            ProposalAttrs m_attrs;
        };
    }
}
