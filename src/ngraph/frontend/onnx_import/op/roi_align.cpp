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

#include <memory>

#include "ngraph/opsets/opset2.hpp"
#include "roi_align.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector roi_align(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const auto rois = node.get_ng_inputs().at(1);
                    const auto num_rois = node.get_ng_inputs().at(2);

                    const auto pooled_h = node.get_attribute_value<int64_t>("output_height", 1);
                    const auto pooled_w = node.get_attribute_value<int64_t>("output_width", 1);
                    const auto sampling_ratio =
                        node.get_attribute_value<int64_t>("sampling_ratio", 1);
                    const auto spatial_scale =
                        node.get_attribute_value<float>("spatial_scale", 1.0f);
                    const auto mode = node.get_attribute_value<std::string>("mode", "avg");

                    return {std::make_shared<ngraph::opset2::ROIAlign>(data,
                                                                       rois,
                                                                       num_rois,
                                                                       pooled_h,
                                                                       pooled_w,
                                                                       sampling_ratio,
                                                                       spatial_scale,
                                                                       mode)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
