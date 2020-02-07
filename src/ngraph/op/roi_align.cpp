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

#include "ngraph/op/roi_align.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v2::ROIAlign::type_info;

shared_ptr<Node> op::v2::ROIAlign::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ROIAlign>(new_args.at(0), new_args.at(1), new_args.at(2), m_pooled_h, m_pooled_w,
            m_sampling_ratio, m_spatial_scale, m_mode);
}

op::v2::ROIAlign::ROIAlign(const Output<Node>& data,
                           const Output<Node>& rois,
                           const Output<Node>& batch_indices,
                           const size_t pooled_h,
                           const size_t pooled_w,
                           const int32_t sampling_ratio,
                           const float spatial_scale,
                           const std::string& mode)
        : Op({data, rois, batch_indices}), m_pooled_h(pooled_h), m_pooled_w(pooled_w), m_sampling_ratio(sampling_ratio),
        m_spatial_scale(spatial_scale), m_mode(mode)
{
    constructor_validate_and_infer_types();
}

void op::v2::ROIAlign::validate_and_infer_types()
{
    const PartialShape& data_shape = get_input_partial_shape(0);
    const PartialShape& rois_shape = get_input_partial_shape(1);
    const PartialShape& batch_indices_shape = get_input_partial_shape(2);

    element::Type data_batch_et = get_input_element_type(0);
    element::Type rois_et = get_input_element_type(1);
    element::Type batch_indices_et = get_input_element_type(2);

    if (data_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(
                this,
                static_cast<size_t>(data_shape.rank()) == 2,
                "The feature map tensor rank is expected to be 4, got: ",
                data_shape.rank());
        NODE_VALIDATION_CHECK(
                this,
                static_cast<size_t>(data_shape[0]) == 1,
                "The feature map tensor batch dimension is expected to be 1, got: ",
                data_shape[0]);
    }

    if (rois_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(
                this,
                static_cast<size_t>(rois_shape.rank()) == 2,
                "The ROIs tensor rank is expected to be 2, got: ",
                rois_shape.rank());
        NODE_VALIDATION_CHECK(
                this,
                static_cast<size_t>(rois_shape[1]) == 4,
                "The ROIs tensor last dimension is expected to be 4, got: ",
                rois_shape[1]);
    }

    if (batch_indices_shape.is_static())
    {
        NODE_VALIDATION_CHECK(
                this,
                static_cast<size_t>(batch_indices_shape.rank()) == 1,
                "The batch indices tensor rank is expected to be 1, got: ",
                batch_indices_shape.rank());
    }

    if (rois_shape.rank().is_static() and batch_indices_shape.is_static() and rois_shape[0].is_static() and
        batch_indices_shape[0].is_static())
    {
        NODE_VALIDATION_CHECK(
                this,
                static_cast<size_t>(batch_indices_shape[0]) == static_cast<size_t>(rois_shape[0]),
                "The number of elements in batch indices tensor and ROIs tensor is expected to be equal, got: ",
                batch_indices_shape[0] , " and ", rois_shape[0]);
    }

    PartialShape result_shape;
    if (rois_shape.is_static())
    {
        result_shape = PartialShape({rois_shape[0], data_shape[1], m_pooled_h, m_pooled_w});
    }
    else
    {
        result_shape = PartialShape({Dimension::dynamic(), data_shape[1], m_pooled_h, m_pooled_w});
    }
    set_output_type(0, data_batch_et, result_shape);
}
