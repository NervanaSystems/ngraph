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

#include "deformable_psroi_pooling.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::DeformablePSROIPooling::type_info;

op::v1::DeformablePSROIPooling::DeformablePSROIPooling(const Output<Node>& input,
                                                       const Output<Node>& coords,
                                                       const Output<Node>& offsets,
                                                       const int64_t output_dim,
                                                       const float spatial_scale,
                                                       const int64_t group_size,
                                                       const std::string mode,
                                                       int64_t spatial_bins_x,
                                                       int64_t spatial_bins_y,
                                                       float trans_std,
                                                       int64_t part_size)
    : Op({input, coords, offsets})
    , m_output_dim(output_dim)
    , m_spatial_scale(spatial_scale)
    , m_group_size(group_size)
    , m_mode(mode)
    , m_spatial_bins_x(spatial_bins_x)
    , m_spatial_bins_y(spatial_bins_y)
    , m_trans_std(trans_std)
    , m_part_size(part_size)
{
    constructor_validate_and_infer_types();
}

op::v1::DeformablePSROIPooling::DeformablePSROIPooling(const Output<Node>& input,
                                                       const Output<Node>& coords,
                                                       const int64_t output_dim,
                                                       const float spatial_scale,
                                                       const int64_t group_size,
                                                       const std::string mode,
                                                       int64_t spatial_bins_x,
                                                       int64_t spatial_bins_y,
                                                       float trans_std,
                                                       int64_t part_size)
    : Op({input, coords})
    , m_output_dim(output_dim)
    , m_spatial_scale(spatial_scale)
    , m_group_size(group_size)
    , m_mode(mode)
    , m_spatial_bins_x(spatial_bins_x)
    , m_spatial_bins_y(spatial_bins_y)
    , m_trans_std(trans_std)
    , m_part_size(part_size)
{
    constructor_validate_and_infer_types();
}

bool op::v1::DeformablePSROIPooling::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("output_dim", m_output_dim);
    visitor.on_attribute("spatial_scale", m_spatial_scale);
    visitor.on_attribute("group_size", m_group_size);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("spatial_bins_x", m_spatial_bins_x);
    visitor.on_attribute("spatial_bins_y", m_spatial_bins_y);
    visitor.on_attribute("trans_std", m_trans_std);
    visitor.on_attribute("part_size", m_part_size);
    return true;
}

void op::v1::DeformablePSROIPooling::validate_and_infer_types()
{
    const auto& input_et = get_input_element_type(0);

    const auto& input_pshape = get_input_partial_shape(0);
    const auto& box_coords_pshape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
                          input_pshape.rank().is_dynamic() || input_pshape.rank().get_length() == 4,
                          "Feature map input rank must equal to 4 (input rank: ",
                          input_pshape.rank().get_length(),
                          ")");
    NODE_VALIDATION_CHECK(this,
                          box_coords_pshape.rank().is_dynamic() ||
                              box_coords_pshape.rank().get_length() == 2,
                          "Box coordinates input rank must equal to 2 (input rank: ",
                          box_coords_pshape.rank().get_length(),
                          ")");

    if (get_input_size() == 3) // offsets input is provided
    {
        const auto& offsets_pshape = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              offsets_pshape.rank().is_dynamic() ||
                                  offsets_pshape.rank().get_length() == 4,
                              "Offsets input rank must equal to 4 (input rank: ",
                              offsets_pshape.rank().get_length(),
                              ")");
    }
    int64_t output_rank = 4;
    std::vector<Dimension> output_dim_vec(output_rank, Dimension::dynamic());
    if (box_coords_pshape[0].is_static())
    {
        output_dim_vec[0] = box_coords_pshape.to_shape()[0];
    }
    output_dim_vec[1] = m_output_dim;
    for (int i = 2; i < output_rank; ++i)
    {
        output_dim_vec[i] = m_group_size;
    }

    set_output_type(0, input_et, PartialShape(output_dim_vec));
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
                                                       m_spatial_scale,
                                                       m_group_size,
                                                       m_mode,
                                                       m_spatial_bins_x,
                                                       m_spatial_bins_y,
                                                       m_trans_std,
                                                       m_part_size);
    }
    else if (new_args.size() == 2)
    {
        return make_shared<v1::DeformablePSROIPooling>(new_args.at(0),
                                                       new_args.at(1),
                                                       m_output_dim,
                                                       m_spatial_scale,
                                                       m_group_size,
                                                       m_mode,
                                                       m_spatial_bins_x,
                                                       m_spatial_bins_y,
                                                       m_trans_std,
                                                       m_part_size);
    }
    else
    {
        throw ngraph_error("Not supported number of DeformablePSROIPooling args");
    }
}
