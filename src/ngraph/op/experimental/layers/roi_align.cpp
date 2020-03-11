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
                           const Output<Node>& batch_indices,
                           const size_t pooled_h,
                           const size_t pooled_w,
                           const size_t sampling_ratio,
                           const float spatial_scale,
                           const std::string& mode)
    : Op{{input, rois, batch_indices}}
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
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0) == element::f32 &&
                              get_input_element_type(1) == element::f32,
                          "The data type for input and ROIs is expected to be float32. Got: ",
                          get_input_element_type(0),
                          " and: ",
                          get_input_element_type(1));

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2) == element::i32,
                          "The data type for batch indices is expected to be int32. Got: ",
                          get_input_element_type(2));

    const auto& input_ps = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          input_ps.rank().compatible(4),
                          "Expected a 4D tensor for the input data. Got: ",
                          input_ps);

    const auto& rois_ps = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          rois_ps.rank().compatible(2),
                          "Expected a 2D tensor for the ROIs input. Got: ",
                          rois_ps);

    const auto rois_second_dim = rois_ps[1];
    NODE_VALIDATION_CHECK(this,
                          rois_second_dim.compatible(4),
                          "The second dimension of ROIs input should contain box coordinates. ",
                          "This dimension is expected to be equal to 4. Got: ",
                          rois_second_dim);

    const auto& batch_indices_ps = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this,
                          batch_indices_ps.rank().compatible(2),
                          "Expected a 2D tensor for the batch indices input. Got: ",
                          batch_indices_ps);

    NODE_VALIDATION_CHECK(this,
                          rois_ps[0].same_scheme(batch_indices_ps[0]),
                          "The first dimension of ROIs input must be equal to the first dimension ",
                          "if the batch indices input. Got: ",
                          rois_ps[0],
                          " and: ",
                          batch_indices_ps[0]);

    NODE_VALIDATION_CHECK(this,
                          m_mode == "avg" || m_mode == "max",
                          "The ROIAlign supports 'avg' and 'max' modes. Got: ",
                          m_mode);

    // the output shape should have the following format [NUM_ROIS, C, pooled_h, pooled_w]
    auto output_shape = PartialShape{{Dimension::dynamic(),
                                      input_ps[1],
                                      Dimension{static_cast<int64_t>(m_pooled_h)},
                                      Dimension{static_cast<int64_t>(m_pooled_w)}}};

    // if either of those 2 dimensions is static its value will be used
    // for the first dimension of the output shape - 'NUM_ROIS'
    if (rois_ps[0].is_static())
    {
        output_shape[0] = rois_ps[0];
    }

    if (batch_indices_ps[0].is_static())
    {
        output_shape[0] = batch_indices_ps[0];
    }

    set_output_size(1);
    set_output_type(0, get_input_element_type(0), output_shape);

    // if the channels dimension is not known
    // the first input should be used during the function specialization
    if (input_ps[1].is_dynamic())
    {
        set_input_is_relevant_to_shape(0);
    }

    // if the 'NUM_ROIS' value is not known
    // the last 2 inputs should be used during the function specialization
    if (output_shape[0].is_dynamic())
    {
        set_input_is_relevant_to_shape(1);
        set_input_is_relevant_to_shape(2);
    }
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
                                 new_args.at(2),
                                 m_pooled_h,
                                 m_pooled_w,
                                 m_sampling_ratio,
                                 m_spatial_scale,
                                 m_mode);
}
