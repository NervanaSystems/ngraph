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

#include "deformable_psroi_pooling.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::DeformablePSROIPooling::type_info;

op::v1::DeformablePSROIPooling::DeformablePSROIPooling(const Output<Node>& input,
                                                       const Output<Node>& coords,
                                                       const Output<Node>& offsets,
                                                       const int64_t output_dim,
                                                       const int64_t group_size,
                                                       float spatial_scale,
                                                       const std::string mode,
                                                       int64_t spatial_bins_x,
                                                       int64_t spatial_bins_y,
                                                       bool no_trans,
                                                       float trans_std,
                                                       int64_t part_size)
    : Op({input, coords, offsets})
    , m_output_dim(output_dim)
    , m_group_size(group_size)
    , m_spatial_scale(spatial_scale)
    , m_mode(mode)
    , m_spatial_bins_x(spatial_bins_x)
    , m_spatial_bins_y(spatial_bins_y)
    , m_no_trans(no_trans)
    , m_trans_std(trans_std)
    , m_part_size(part_size)
{
    constructor_validate_and_infer_types();
}

op::v1::DeformablePSROIPooling::DeformablePSROIPooling(const Output<Node>& input,
                                                       const Output<Node>& coords,
                                                       const int64_t output_dim,
                                                       const int64_t group_size,
                                                       float spatial_scale,
                                                       const std::string mode,
                                                       int64_t spatial_bins_x,
                                                       int64_t spatial_bins_y,
                                                       bool no_trans,
                                                       float trans_std,
                                                       int64_t part_size)
    : Op({input, coords})
    , m_output_dim(output_dim)
    , m_group_size(group_size)
    , m_spatial_scale(spatial_scale)
    , m_mode(mode)
    , m_spatial_bins_x(spatial_bins_x)
    , m_spatial_bins_y(spatial_bins_y)
    , m_no_trans(no_trans)
    , m_trans_std(trans_std)
    , m_part_size(part_size)
{
    constructor_validate_and_infer_types();
}

void op::v1::DeformablePSROIPooling::validate_and_infer_types()
{
    auto input_et = get_input_element_type(0);
    if (get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static())
    {
        Shape input_shape = get_input_partial_shape(0).to_shape();
        Shape coords_shape = get_input_partial_shape(1).to_shape();
        NODE_VALIDATION_CHECK(
            this,
            input_shape.size() >= 3,
            "DeformablePSROIPooling expects 3 or higher dimensions for input. Got ",
            input_shape.size());
        NODE_VALIDATION_CHECK(
            this,
            coords_shape.size() == 2,
            "DeformablePSROIPooling expects 2 dimensions for box coordinates. Got ",
            coords_shape.size());
        Shape output_shape{coords_shape[0], m_output_dim};
        for (size_t i = 2; i < input_shape.size(); i++)
        {
            output_shape.push_back(m_group_size);
        }
        set_output_type(0, input_et, output_shape);
    }
    else
    {
        set_output_type(0, input_et, PartialShape::dynamic());
    }
}

shared_ptr<Node>
    op::v1::DeformablePSROIPooling::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 3)
    {
        return make_shared<v1::DeformablePSROIPooling>(new_args.at(0),
                                                       new_args.at(1),
                                                       new_args.at(2),
                                                       m_output_dim,
                                                       m_group_size,
                                                       m_spatial_scale,
                                                       m_mode,
                                                       m_spatial_bins_x,
                                                       m_spatial_bins_y,
                                                       m_no_trans,
                                                       m_trans_std,
                                                       m_part_size);
    }
    else if (new_args.size() == 2)
    {
        return make_shared<v1::DeformablePSROIPooling>(new_args.at(0),
                                                       new_args.at(1),
                                                       m_output_dim,
                                                       m_group_size,
                                                       m_spatial_scale,
                                                       m_mode,
                                                       m_spatial_bins_x,
                                                       m_spatial_bins_y,
                                                       m_no_trans,
                                                       m_trans_std,
                                                       m_part_size);
    }
    else
    {
        throw ngraph_error("Not supported number of DeformablePSROIPooling args");
    }
}
