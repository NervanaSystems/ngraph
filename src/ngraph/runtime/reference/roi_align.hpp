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
//****************************************************************************

#pragma once

#include "ngraph/op/roi_align.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            inline void roi_align(T* in, // [N, C, H, W]
                                  T* boxes, // each row [x_1, y_1, x_2, y_2]
                                  T* batch_indices,
                                  T* out,
                                  const int pooled_h,
                                  const int pooled_w,
                                  const int sampling_ratio,
                                  const float spatial_scale,
                                  const op::v3::ROIAlign::PoolingMode& mode)
            {
                // x1 = boxes[0] * spatial_scale
                // y1 = boxes[1] * spatial_scale
                // x2 = boxes[2] * spatial_scale
                // y2 = boxes[3] * spatial_scale
                // roi_w = max(x2 - x1, 1);
                // roi_h = max(y2 - y1, 1);
                // bin_w = roi_w / pooled_w;
                // bin_h = roi_h / pooled_h;

                // sampling_w = sampling_ratio * pooled_w;
                // sampling_h = sampling_ratio * pooled_h;
                
                // sampling_x1 = x1 + 0.5 * bin_w / sampling_ratio;
                // sampling_y1 = y1 + 0.5 * bin_h / sampling_ratio;
                // sampling_x2 = x2 - 0.5 * bin_w / sampling_ratio;
                // sampling_y2 = y2 - 0.5 * bin_h / sampling_ratio;
                // sampling_boxes = [sampling_x1, sampling_y1,
                //                   sampling_x2, sampling_y2];
                // samples = crop_and_resize(in, sampling_boxes, batch_indices, 
                //                 shape{sampling_h, sampling_w}, ResizeMthod::bilinear);

                // do pooling with no padding
                // switch(mode)
                // case(PoolingMode::max)
                //     out = max_pool(samples, Shape{sampling_ratio, sampling_ratio});
                // case(PoolingMode::avg)
                //     out = avg_pool(samples, Shape{sampling_ratio, sampling_ratio});
                


            }
                                
                                  
        }
    }
}