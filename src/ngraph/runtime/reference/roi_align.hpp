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
            inline void roi_align(T* in,
                                  T* boxes,
                                  T* boxes_indices,
                                  T* out,
                                  const int pooled_h,
                                  const int pooled_w,
                                  const int sampling_ratio,
                                  const float spatial_scale,
                                  const op::v3::ROIAlign::PoolingMode& mode)
            {

            }
                                
                                  
        }
    }
}