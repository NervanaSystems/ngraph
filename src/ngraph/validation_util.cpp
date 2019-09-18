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

#include "ngraph/validation_util.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

Strides ngraph::conv_default_strides(const Node* /* node */,
                                     const PartialShape& data_batch_shape,
                                     const PartialShape& filters_shape)
{
    size_t rank;

    if (data_batch_shape.rank().is_static() && static_cast<size_t>(data_batch_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(data_batch_shape.rank()) - 2;
    }
    else if (filters_shape.rank().is_static() && static_cast<size_t>(filters_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(filters_shape.rank()) - 2;
    }
    else
    {
        rank = 0;
    }

    return Strides(rank, 1);
}

CoordinateDiff ngraph::conv_default_padding(const Node* /* node */,
                                            const PartialShape& data_batch_shape,
                                            const PartialShape& filters_shape)
{
    size_t rank;

    if (data_batch_shape.rank().is_static() && static_cast<size_t>(data_batch_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(data_batch_shape.rank()) - 2;
    }
    else if (filters_shape.rank().is_static() && static_cast<size_t>(filters_shape.rank()) >= 2)
    {
        rank = static_cast<size_t>(filters_shape.rank()) - 2;
    }
    else
    {
        rank = 0;
    }

    return CoordinateDiff(rank, 0);
}

//
// Infers the output shape of a windowed reduction operation, where the data may be dilated and/or
// padded, and the reduction window may be strided and/or dilated.
//
// TODO(amprocte): The messages here would be a bit friendlier if we didn't say "after
// padding/after dilation" for cases where there is actually no padding/dilation.
//
PartialShape ngraph::infer_windowed_reduction_output_shape(const Node* node,
                                                           const PartialShape& data_shape,
                                                           const Strides& data_dilation,
                                                           const CoordinateDiff& data_padding_below,
                                                           const CoordinateDiff& data_padding_above,
                                                           const PartialShape& window_shape,
                                                           const Strides& window_strides,
                                                           const Strides& window_dilation,
                                                           bool is_window_all_in_padding_allowed,
                                                           bool ceil_mode)
{
    PartialShape data_shape_merged{PartialShape::dynamic()};

    NODE_VALIDATION_CHECK(node,
                          data_shape_merged.merge_rank(data_shape.rank()) &&
                              data_shape_merged.merge_rank(data_dilation.size()) &&
                              data_shape_merged.merge_rank(data_padding_below.size()) &&
                              data_shape_merged.merge_rank(data_padding_above.size()) &&
                              data_shape_merged.merge_rank(window_shape.rank()) &&
                              data_shape_merged.merge_rank(window_strides.size()) &&
                              data_shape_merged.merge_rank(window_dilation.size()),
                          "Ranks for data shape (",
                          data_shape,
                          "), data dilation (",
                          data_dilation,
                          "), padding below (",
                          data_padding_below,
                          "), padding above (",
                          data_padding_above,
                          "), window shape (",
                          window_shape,
                          "), window strides (",
                          window_strides,
                          "), and window dilation (",
                          window_dilation,
                          ") do not match.");

    PartialShape output_shape = PartialShape::dynamic(data_shape_merged.rank());

    if (output_shape.rank().is_static())
    {
        for (size_t i = 0; i < static_cast<size_t>(output_shape.rank()); i++)
        {
            NODE_VALIDATION_CHECK(node,
                                  data_dilation[i] > 0,
                                  "Data dilation (",
                                  data_dilation,
                                  ") has zero dimension at axis ",
                                  i,
                                  ".");
            NODE_VALIDATION_CHECK(node,
                                  window_strides[i] > 0,
                                  "Window strides (",
                                  window_strides,
                                  ") has zero dimension at axis ",
                                  i,
                                  ".");
            NODE_VALIDATION_CHECK(node,
                                  window_dilation[i] > 0,
                                  "Window dilation (",
                                  window_dilation,
                                  ") has zero dimension at axis ",
                                  i,
                                  ".");

            bool data_dim_static = data_shape.rank().is_static() && data_shape[i].is_static();
            bool window_dim_static = window_shape.rank().is_static() && window_shape[i].is_static();

            ptrdiff_t data_padded_dilated_dim = -1;
            if (data_dim_static)
            {
                data_padded_dilated_dim = (static_cast<int64_t>(data_dilation[i]) *
                                           (static_cast<int64_t>(data_shape[i]) - 1)) +
                                          1 + data_padding_below[i] + data_padding_above[i];
                NODE_VALIDATION_CHECK(
                    node,
                    data_padded_dilated_dim > 0,
                    "Data shape after padding and dilation has dimension less than 1 (dim: ",
                    data_padded_dilated_dim,
                    ") at axis ",
                    i,
                    ".");
            }

            ptrdiff_t window_dilated_dim = -1;
            if (window_dim_static)
            {
                window_dilated_dim = static_cast<int64_t>(window_dilation[i]) *
                                         (static_cast<int64_t>(window_shape[i]) - 1) +
                                     1;

                NODE_VALIDATION_CHECK(node,
                                      window_dilated_dim > 0,
                                      "Window after dilation has dimension less than 1 (dim: ",
                                      window_dilated_dim,
                                      ") at axis ",
                                      i,
                                      ".");

                NODE_VALIDATION_CHECK(
                    node,
                    is_window_all_in_padding_allowed ||
                        (window_dilated_dim > data_padding_below[i] &&
                         window_dilated_dim > data_padding_above[i]),
                    "Window after dilation is sometimes entirely in the padding area for axis ",
                    i,
                    " (dilated window dimension: ",
                    window_dilated_dim,
                    ", padding below dimension: ",
                    data_padding_below[i],
                    ", padding above dimension: ",
                    data_padding_above[i],
                    ") and this is not ",
                    "allowed.");
            }

            if (data_dim_static && window_dim_static)
            {
                NODE_VALIDATION_CHECK(node,
                                      window_dilated_dim <= data_padded_dilated_dim,
                                      "Window after dilation has dimension (dim: ",
                                      window_dilated_dim,
                                      ") larger than the data shape after padding (dim: ",
                                      data_padded_dilated_dim,
                                      ") at axis ",
                                      i,
                                      ".");

                if (ceil_mode)
                {
                    output_shape[i] = ceil_div(static_cast<size_t>(data_padded_dilated_dim) -
                                                   static_cast<size_t>(window_dilated_dim),
                                               window_strides[i]) +
                                      1;
                }
                else
                {
                    output_shape[i] = ((static_cast<size_t>(data_padded_dilated_dim) -
                                        static_cast<size_t>(window_dilated_dim)) /
                                       window_strides[i]) +
                                      1;
                }
            }
        }
    }

    return output_shape;
}

//
// Infers the output batch shape and element type for convolution fprop.
//
PartialShape ngraph::infer_convolution_forward(const Node* node,
                                               const PartialShape& data_batch_shape,
                                               const Strides& data_dilation,
                                               const CoordinateDiff& data_padding_below,
                                               const CoordinateDiff& data_padding_above,
                                               const PartialShape& filters_shape,
                                               const Strides& filter_strides,
                                               const Strides& filter_dilation)
{
    Rank data_batch_filters_rank{Rank::dynamic()};

    NODE_VALIDATION_CHECK(
        node,
        Rank::merge(data_batch_filters_rank, data_batch_shape.rank(), filters_shape.rank()),
        "Data batch and filters rank do not match (data batch shape: ",
        data_batch_shape,
        ", filters shape: ",
        filters_shape,
        ").");

    NODE_VALIDATION_CHECK(node,
                          data_batch_filters_rank.is_dynamic() ||
                              static_cast<size_t>(data_batch_filters_rank) >= 3,
                          "Data batch and filters must have rank of at least 3 (one batch axis, ",
                          "one input-channel axis, and at least one spatial dimension) ",
                          "(data batch shape: ",
                          data_batch_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");

    Rank spatial_rank{Rank::dynamic()};
    NODE_VALIDATION_CHECK(node,
                          Rank::merge(spatial_rank, spatial_rank, data_batch_filters_rank - 2) &&
                              Rank::merge(spatial_rank, spatial_rank, data_dilation.size()) &&
                              Rank::merge(spatial_rank, spatial_rank, data_padding_below.size()) &&
                              Rank::merge(spatial_rank, spatial_rank, data_padding_above.size()) &&
                              Rank::merge(spatial_rank, spatial_rank, filter_strides.size()) &&
                              Rank::merge(spatial_rank, spatial_rank, filter_dilation.size()),
                          "Ranks for data item shape/filters shape (data batch has shape ",
                          data_batch_shape,
                          ", so data item rank is ",
                          (data_batch_shape.rank() - 2),
                          " and filters have shape ",
                          filters_shape,
                          ", so filters spatial rank is ",
                          (filters_shape.rank() - 2),
                          "), data dilation (",
                          data_dilation,
                          "), padding below (",
                          data_padding_below,
                          "), padding above (",
                          data_padding_above,
                          "), filter strides (",
                          filter_strides,
                          "), and filter dilation (",
                          filter_dilation,
                          ") do not match.");

    Dimension batch_size =
        (data_batch_shape.rank().is_static() ? data_batch_shape[0] : Dimension::dynamic());
    Dimension data_channel_count =
        (data_batch_shape.rank().is_static() ? data_batch_shape[1] : Dimension::dynamic());
    PartialShape data_spatial_shape(PartialShape::dynamic(spatial_rank));

    Dimension filter_output_channel_count =
        (filters_shape.rank().is_static() ? filters_shape[0] : Dimension::dynamic());
    Dimension filter_input_channel_count =
        (filters_shape.rank().is_static() ? filters_shape[1] : Dimension::dynamic());
    PartialShape filter_spatial_shape(PartialShape::dynamic(spatial_rank));

    //
    // Note: spatial_rank is definitely static at this point.
    //

    for (size_t i = 0; i < static_cast<size_t>(spatial_rank); i++)
    {
        if (data_batch_shape.rank().is_static())
        {
            data_spatial_shape[i] = data_batch_shape[i + 2];
        }

        if (filters_shape.rank().is_static())
        {
            filter_spatial_shape[i] = filters_shape[i + 2];
        }
    }

    NODE_VALIDATION_CHECK(node,
                          batch_size.is_dynamic() || static_cast<size_t>(batch_size) > 0,
                          "Batch size is zero.");

    Dimension merged_channel_count;

    NODE_VALIDATION_CHECK(
        node,
        Dimension::merge(merged_channel_count, data_channel_count, filter_input_channel_count),
        "Data batch channel count (",
        data_channel_count,
        ") does not match filter input ",
        "channel count (",
        filter_input_channel_count,
        ").");

    NODE_VALIDATION_CHECK(node,
                          merged_channel_count.is_dynamic() ||
                              static_cast<size_t>(merged_channel_count) > 0,
                          "Data batch channel count and/or filter input channel count is zero.");

    NODE_VALIDATION_CHECK(node,
                          filter_output_channel_count.is_dynamic() ||
                              static_cast<size_t>(filter_output_channel_count) > 0,
                          "Filter output channel count is zero.");

    PartialShape data_output_shape = infer_windowed_reduction_output_shape(node,
                                                                           data_spatial_shape,
                                                                           data_dilation,
                                                                           data_padding_below,
                                                                           data_padding_above,
                                                                           filter_spatial_shape,
                                                                           filter_strides,
                                                                           filter_dilation,
                                                                           true);

    PartialShape batch_output_shape(PartialShape::dynamic(spatial_rank + 2));
    batch_output_shape[0] = batch_size;
    batch_output_shape[1] = filter_output_channel_count;

    for (size_t i = 0; i < static_cast<size_t>(spatial_rank); i++)
    {
        batch_output_shape[i + 2] = data_output_shape[i];
    }

    return batch_output_shape;
}

//
// Infers the output batch shape and element type for batched pooling fprop.
//
PartialShape ngraph::infer_batched_pooling_forward(const Node* node,
                                                   const PartialShape& data_batch_shape,
                                                   const CoordinateDiff& data_padding_below,
                                                   const CoordinateDiff& data_padding_above,
                                                   const PartialShape& window_shape,
                                                   const Strides& window_strides,
                                                   bool is_window_all_in_padding_allowed,
                                                   bool ceil_mode)
{
    NODE_VALIDATION_CHECK(node,
                          data_batch_shape.rank().is_dynamic() ||
                              static_cast<size_t>(data_batch_shape.rank()) >= 3,
                          "Data batch must have rank of at least 3 (one batch axis, ",
                          "one input-channel axis, and at least one spatial dimension) ",
                          "(data batch shape: ",
                          data_batch_shape,
                          ").");

    PartialShape data_spatial_shape{PartialShape::dynamic()};

    NODE_VALIDATION_CHECK(node,
                          data_spatial_shape.merge_rank(data_batch_shape.rank() - 2) &&
                              data_spatial_shape.merge_rank(data_padding_below.size()) &&
                              data_spatial_shape.merge_rank(data_padding_above.size()) &&
                              data_spatial_shape.merge_rank(window_shape.rank()) &&
                              data_spatial_shape.merge_rank(window_strides.size()),
                          "Ranks for data item shape (data batch has shape ",
                          data_batch_shape,
                          ", so data item rank is ",
                          (data_batch_shape.rank() - 2),
                          "), padding below (",
                          data_padding_below,
                          "), padding above (",
                          data_padding_above,
                          "), window shape (",
                          window_shape,
                          "), and window strides (",
                          window_strides,
                          ") do not match.");

    Dimension batch_size{Dimension::dynamic()};
    Dimension channel_count{Dimension::dynamic()};
    PartialShape data_output_spatial_shape{PartialShape::dynamic(data_spatial_shape.rank())};

    if (data_batch_shape.rank().is_static())
    {
        batch_size = data_batch_shape[0];
        channel_count = data_batch_shape[1];

        for (size_t i = 0; i < static_cast<size_t>(data_spatial_shape.rank()); i++)
        {
            data_spatial_shape[i] = data_batch_shape[i + 2];
        }

        NODE_VALIDATION_CHECK(node,
                              batch_size.is_dynamic() || static_cast<size_t>(batch_size) > 0,
                              "Batch size is zero.");

        NODE_VALIDATION_CHECK(node,
                              channel_count.is_dynamic() || static_cast<size_t>(channel_count) > 0,
                              "Channel count is zero.");

        // For pooling ops we don't need dilation, so we fill in the identity value (all 1).
        Strides data_dilation(static_cast<size_t>(data_spatial_shape.rank()), 1);
        Strides window_dilation(static_cast<size_t>(data_spatial_shape.rank()), 1);

        data_output_spatial_shape =
            infer_windowed_reduction_output_shape(node,
                                                  data_spatial_shape,
                                                  data_dilation,
                                                  data_padding_below,
                                                  data_padding_above,
                                                  window_shape,
                                                  window_strides,
                                                  window_dilation,
                                                  is_window_all_in_padding_allowed,
                                                  ceil_mode);
    }

    PartialShape data_batch_output_shape{
        PartialShape::dynamic(data_output_spatial_shape.rank() + 2)};
    data_batch_output_shape[0] = batch_size;
    data_batch_output_shape[1] = channel_count;

    for (size_t i = 0; i < static_cast<size_t>(data_spatial_shape.rank()); i++)
    {
        data_batch_output_shape[i + 2] = data_output_spatial_shape[i];
    }

    return data_batch_output_shape;
}

struct ChannelShapedInputSpec
{
    element::Type m_element_type;
    PartialShape m_shape;
    std::string m_input_name;
};

static std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward_helper(
    const Node* node,
    element::Type input_element_type,
    const PartialShape& input_shape,
    const std::vector<ChannelShapedInputSpec>& channel_shaped_inputs)
{
    // Built up a slash-separated string naming all the channel-shaped inputs, for use in error
    // messages.
    std::stringstream ss;
    bool first = true;
    for (auto& inp : channel_shaped_inputs)
    {
        if (!first)
        {
            ss << "/";
        }
        ss << inp.m_input_name;
        first = false;
    }
    std::string channel_input_names = ss.str();

    // Infer output element type.
    element::Type et_result{input_element_type};

    for (auto& inp : channel_shaped_inputs)
    {
        NODE_VALIDATION_CHECK(node,
                              element::Type::merge(et_result, et_result, inp.m_element_type),
                              "Input element types do not match.");
    }

    // Extract channel dimension from input shape.
    Dimension channel_dim{Dimension::dynamic()};

    NODE_VALIDATION_CHECK(node,
                          input_shape.is_dynamic() || static_cast<size_t>(input_shape.rank()) >= 2,
                          "Input argument must have rank of at least 2 (input argument shape: ",
                          input_shape,
                          ").");

    if (input_shape.rank().is_static())
    {
        channel_dim = input_shape[1];
    }

    // Infer gamma/beta/mu/sigma shape, which must be consistent with a vector of size
    // "channel_dim".
    PartialShape channel_shape{PartialShape::dynamic()};

    for (auto& inp : channel_shaped_inputs)
    {
        NODE_VALIDATION_CHECK(node,
                              PartialShape::merge_into(channel_shape, inp.m_shape),
                              "Shapes for ",
                              channel_input_names,
                              " do not match.");
    }

    NODE_VALIDATION_CHECK(node,
                          channel_shape.merge_rank(1),
                          "Shape for ",
                          channel_input_names,
                          " (",
                          channel_shape,
                          ") does not have rank 1.");

    NODE_VALIDATION_CHECK(node,
                          Dimension::merge(channel_dim, channel_dim, channel_shape[0]),
                          "Input channel dimension (",
                          channel_dim,
                          ") does not match shape for ",
                          channel_input_names,
                          " (",
                          channel_shape,
                          ").");

    NODE_VALIDATION_CHECK(node,
                          channel_dim.is_dynamic() || static_cast<size_t>(channel_dim) >= 1,
                          "Channel count must be at least 1.");

    // Batch result shape is same as the input shape, except we may possibly have inferred more
    // information from the channel count via gamma/beta/etc.
    PartialShape batch_result_shape{input_shape};

    if (batch_result_shape.rank().is_static())
    {
        batch_result_shape[1] = channel_dim;
    }

    return std::make_tuple(et_result, batch_result_shape, PartialShape{channel_dim});
}

std::tuple<element::Type, PartialShape, PartialShape>
    ngraph::infer_batch_norm_forward(const Node* node,
                                     element::Type input_element_type,
                                     element::Type gamma_element_type,
                                     element::Type beta_element_type,
                                     element::Type mean_element_type,
                                     element::Type variance_element_type,
                                     const PartialShape& input_shape,
                                     const PartialShape& gamma_shape,
                                     const PartialShape& beta_shape,
                                     const PartialShape& mean_shape,
                                     const PartialShape& variance_shape)
{
    return infer_batch_norm_forward_helper(node,
                                           input_element_type,
                                           input_shape,
                                           {{gamma_element_type, gamma_shape, "gamma"},
                                            {beta_element_type, beta_shape, "beta"},
                                            {mean_element_type, mean_shape, "mean"},
                                            {variance_element_type, variance_shape, "variance"}});
}

std::tuple<element::Type, PartialShape, PartialShape>
    ngraph::infer_batch_norm_forward(const Node* node,
                                     element::Type input_element_type,
                                     element::Type gamma_element_type,
                                     element::Type beta_element_type,
                                     const PartialShape& input_shape,
                                     const PartialShape& gamma_shape,
                                     const PartialShape& beta_shape)
{
    return infer_batch_norm_forward_helper(
        node,
        input_element_type,
        input_shape,
        {{gamma_element_type, gamma_shape, "gamma"}, {beta_element_type, beta_shape, "beta"}});
}

void ngraph::infer_auto_padding(const Shape& image_shape,
                                const Shape& filter_shape,
                                const Strides& filter_strides,
                                const Strides& filter_dilations,
                                const op::PadType pad_type,
                                CoordinateDiff& padding_above,
                                CoordinateDiff& padding_below)
{
    NGRAPH_CHECK(pad_type == op::PadType::SAME_UPPER || pad_type == op::PadType::SAME_LOWER);
    for (size_t i = 0; i < static_cast<size_t>(filter_shape.size()); i++)
    {
        int64_t image_size = static_cast<int64_t>(image_shape[i + 2]);
        int64_t filter_size = (static_cast<int64_t>(filter_shape[i]) - 1) * filter_dilations[i] + 1;
        int64_t filter_stride = static_cast<int64_t>(filter_strides[i]);
        auto output_size = (image_size + filter_stride - 1) / filter_stride;

        auto padding_needed =
            std::max(int64_t(0), (output_size - 1) * filter_stride + filter_size - image_size);
        auto padding_lhs = padding_needed / 2;
        auto padding_rhs = padding_needed - padding_lhs;
        padding_below.push_back(pad_type == op::PadType::SAME_UPPER ? padding_lhs : padding_rhs);
        padding_above.push_back(pad_type == op::PadType::SAME_UPPER ? padding_rhs : padding_lhs);
    }
}

PartialShape ngraph::infer_slice_shape(const Node* node,
                                       const PartialShape& input_shape,
                                       const std::vector<int64_t>& lb,
                                       const std::vector<int64_t>& ub,
                                       const std::vector<int64_t>& str,
                                       const AxisSet& lb_mask,
                                       const AxisSet& ub_mask,
                                       const AxisSet& new_axis,
                                       const AxisSet& shrink_axis,
                                       const AxisSet& ellipsis_mask)
{
    if (lb.size() && ub.size())
    {
        NODE_VALIDATION_CHECK(node,
                              lb.size() == ub.size(),
                              "Lower bounds and Upper bounds needs to have same number of values");
    }
    if (lb.size() && str.size())
    {
        NODE_VALIDATION_CHECK(node,
                              lb.size() == str.size(),
                              "Lower bounds and strides needs to have same number of values");
    }
    if (ub.size() && str.size())
    {
        NODE_VALIDATION_CHECK(node,
                              ub.size() == str.size(),
                              "Upper bounds and strides needs to have same number of values");
    }

    if (input_shape.rank().is_dynamic())
    {
        return PartialShape::dynamic();
    }

    int max_dims = size_t(input_shape.rank()) + new_axis.size();

    int bounds_size =
        lb.size() ? lb.size() : (ub.size() ? ub.size() : (str.size() ? str.size() : 0));

    int ellipsis_pos1 = ellipsis_mask.size() ? *ellipsis_mask.begin() : max_dims;

    int ellipsis_pos2 = max_dims;
    bounds_size -= ellipsis_pos1;
    if (bounds_size > 0 && (max_dims - bounds_size) > ellipsis_pos1)
    {
        ellipsis_pos2 = max_dims - bounds_size;
    }

    std::vector<Dimension> begin_dms(max_dims, 0);
    std::vector<Dimension> end_dms(max_dims, -1);
    std::vector<Dimension> stride_dms(max_dims, 1);

    std::vector<Dimension> out_dims;

    int j = 0;
    int k = 0;
    int bj = 0;
    int ej = 0;
    int sj = 0;

    for (int i = 0; i < max_dims; i++)
    {
        if (i >= ellipsis_pos1 && i < ellipsis_pos2)
        {
            if (new_axis.find(i) == new_axis.end())
            {
                if (end_dms[i].is_static() && int64_t(end_dms[i]) < 0)
                {
                    end_dms[i] = input_shape[j++] + end_dms[i];
                }
            }
            else
            {
                end_dms[i] = begin_dms[i];
            }

            if (end_dms[i].is_dynamic() || begin_dms[i].is_dynamic() || stride_dms[i].is_dynamic())
            {
                out_dims.push_back(Dimension::dynamic());
            }
            else
            {
                out_dims.push_back(static_cast<int64_t>(
                    ceil(static_cast<float>(abs(int64_t(end_dms[i]) - int64_t(begin_dms[i])) + 1) /
                         static_cast<float>(abs(int64_t(stride_dms[i]))))));
            }
            k = ellipsis_pos1;
            continue;
        }
        stride_dms[i] = (str.size() > sj && str[sj] != 0) ? str[sj++] : 1;

        // Use lower_bounds if mask is not set
        if (lb_mask.find(j) == lb_mask.end())
        {
            if (lb.size() > bj)
            {
                begin_dms[i] = lb[bj];
            }
            else if (stride_dms[i].is_dynamic())
            {
                begin_dms[i] = Dimension::dynamic();
            }
            else if (int64_t(stride_dms[i]) > 0)
            {
                begin_dms[i] = 0;
            }
            else
            {
                begin_dms[i] = -1;
            }
        }
        else if (stride_dms[i].is_dynamic())
        {
            begin_dms[i] = Dimension::dynamic();
        }
        else if (int64_t(stride_dms[i]) > 0)
        {
            begin_dms[i] = 0;
        }
        else
        {
            begin_dms[i] = -1;
        }

        bj++;

        if (begin_dms[i].is_static() && int64_t(begin_dms[i]) < 0)
        {
            begin_dms[i] = input_shape[j] + begin_dms[i];
        }
        // Clipping 'begin'
        if (begin_dms[i].is_static())
        {
            if (int64_t(begin_dms[i]) < 0)
            {
                begin_dms[i] = 0;
            }
            else if (input_shape[j].is_dynamic())
            {
                begin_dms[i] = Dimension::dynamic();
            }
            else if (int64_t(begin_dms[i]) >= int64_t(input_shape[j]))
            {
                begin_dms[i] = input_shape[j] - 1;
            }
        }

        // Use upper_bounds if mask is not set
        if (ub_mask.find(j) == ub_mask.end())
        {
            Dimension end_dms_tmp;

            if (ub.size() <= ej)
            {
                end_dms_tmp = end_dms[i];
            }
            else if (stride_dms[i].is_dynamic())
            {
                end_dms_tmp = Dimension::dynamic();
            }
            else if (int64_t(stride_dms[i]) > 0)
            {
                end_dms_tmp = ub[ej] - 1;
            }
            else
            {
                end_dms_tmp = ub[ej] + 1;
            }

            if (ub.size() > ej)
            {
                end_dms[i] = end_dms_tmp;
            }
            else if (stride_dms[i].is_dynamic())
            {
                end_dms[i] = Dimension::dynamic();
            }
            else if (int64_t(stride_dms[i]) > 0)
            {
                end_dms[i] = -1;
            }
            else
            {
                end_dms[i] = 0;
            }
        }
        else
        {
            if (stride_dms[i].is_dynamic())
            {
                end_dms[i] = Dimension::dynamic();
            }
            else if (int64_t(stride_dms[i]) > 0)
            {
                end_dms[i] = -1;
            }
            else
            {
                end_dms[i] = 0;
            }
        }
        ej++;

        if (end_dms[i].is_static() && int64_t(end_dms[i]) < 0)
        {
            end_dms[i] = input_shape[j] + end_dms[i];
        }
        // Clipping 'end'
        if (end_dms[i].is_static())
        {
            if (int64_t(end_dms[i]) < 0)
            {
                end_dms[i] = 0;
            }
            else if (input_shape[j].is_dynamic())
            {
                end_dms[i] = Dimension::dynamic();
            }
            else if (int64_t(end_dms[i]) >= int64_t(input_shape[j]))
            {
                end_dms[i] = input_shape[j] - 1;
            }
        }

        if (new_axis.find(i) == new_axis.end())
        {
            j++;
        }
        else
        {
            end_dms[i] = 0;
        }

        if (shrink_axis.find(k) != shrink_axis.end())
        {
            end_dms[i] = begin_dms[i];
        }
        else if (end_dms[i].is_dynamic() || begin_dms[i].is_dynamic() || stride_dms[i].is_dynamic())
        {
            out_dims.push_back(Dimension::dynamic());
        }
        else
        {
            out_dims.push_back(static_cast<int64_t>(
                ceil(static_cast<float>(abs(int64_t(end_dms[i]) - int64_t(begin_dms[i])) + 1) /
                     static_cast<float>(abs(int64_t(stride_dms[i]))))));
        }

        k++;
    }
    return out_dims;
}
