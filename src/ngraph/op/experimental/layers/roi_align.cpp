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

#include "roi_align.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::ROIAlign::type_info;

op::v0::ROIAlign::ROIAlign(const Output<Node>& input,
                           const Output<Node>& rois,
                           const size_t pooled_h,
                           const size_t pooled_w,
                           const size_t sampling_ratio,
                           const float spatial_scale,
                           const std::string& mode)
    : Op{{input, rois}}
    , m_pooled_h{pooled_h}
    , m_pooled_w{pooled_w}
    , m_sampling_ratio{sampling_ratio}
    , m_spatial_scale{spatial_scale}
    , m_mode{mode}
{
    constructor_validate_and_infer_types();
}

void op::v0::ROIAlign::validate_and_infer_types()
{
    // TODO
}

bool op::v0::ROIAlign::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("pooled_h", m_pooled_h);
    visitor.on_attribute("pooled_w", m_pooled_w);
    visitor.on_attribute("sampling_ratio", m_sampling_ratio);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("mode", m_mode);

    return true;
}

shared_ptr<Node> op::v0::ROIAlign::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ROIAlign>(new_args.at(0),
                                 new_args.at(1),
                                 m_pooled_h,
                                 m_pooled_w,
                                 m_sampling_ratio,
                                 m_spatial_scale,
                                 m_mode);
}
