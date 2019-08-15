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
        typedef struct
        {
            int num_classes;
            int background_label_id = 0;
            int top_k = -1;
            bool variance_encoded_in_target = false;
            std::vector<int> keep_top_k = {1};
            std::string code_type = std::string{"caffe.PriorBoxParameter.CORNER"};
            bool share_location = true;
            float nms_threshold;
            float confidence_threshold = std::numeric_limits<float>::min();
            bool clip_after_nms = false;
            bool clip_before_nms = false;
            bool decrease_label_id = false;
            bool normalized = false;
            size_t input_height = 1;
            size_t input_width = 1;
            float objectness_score = 0;
        } DetectionOutputAttrs;

        /// \brief Layer which performs non-max suppression to
        /// generate detection output using location and confidence predictions
        class DetectionOutput : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a DetectionOutput operation
            ///
            /// \param box_logits			Box logits
            /// \param class_preds			Class predictions
            /// \param proposals			Proposals
            /// \param aux_class_preds		Auxilary class predictions
            /// \param aux_box_preds		Auxilary box predictions
            /// \param attrs				Detection Output attributes
            DetectionOutput(const std::shared_ptr<Node>& box_logits,
                            const std::shared_ptr<Node>& class_preds,
                            const std::shared_ptr<Node>& proposals,
                            const std::shared_ptr<Node>& aux_class_preds,
                            const std::shared_ptr<Node>& aux_box_preds,
                            const DetectionOutputAttrs& attrs);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const DetectionOutputAttrs& get_attrs() const { return m_attrs; }
        private:
            DetectionOutputAttrs m_attrs;
        };
    }
}
