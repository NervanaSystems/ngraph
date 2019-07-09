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

#include <iterator>
#include <numeric>

#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/fused/group_conv_transpose.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::GroupConvolutionTranspose::GroupConvolutionTranspose(const shared_ptr<Node>& data,
                                                         const shared_ptr<Node>& filters,
                                                         const Strides& strides,
                                                         const Strides& dilations,
                                                         const CoordinateDiff& padding_begin,
                                                         const CoordinateDiff& padding_end,
                                                         const CoordinateDiff& output_padding,
                                                         const size_t groups,
                                                         const PadType& pad_type,
                                                         const Shape& output_shape)
    : FusedOp("GroupConvolutionTranspose", check_single_output_args({data, filters}))
    , m_strides(strides)
    , m_dilations(dilations)
    , m_padding_begin(padding_begin)
    , m_padding_end(padding_end)
    , m_output_padding(output_padding)
    , m_groups(groups)
    , m_pad_type(pad_type)
    , m_output_shape(output_shape)
{
    constructor_validate_and_infer_types();
}

op::GroupConvolutionTranspose::GroupConvolutionTranspose(const std::shared_ptr<Node>& data,
                                                         const std::shared_ptr<Node>& filters,
                                                         const std::size_t groups)
    : GroupConvolutionTranspose(data,
                                filters,
                                Strides(),
                                Strides(),
                                CoordinateDiff(),
                                CoordinateDiff(),
                                CoordinateDiff(),
                                groups,
                                PadType::EXPLICIT,
                                Shape())
{
}

op::GroupConvolutionTranspose::GroupConvolutionTranspose(const std::shared_ptr<Node>& data,
                                                         const std::shared_ptr<Node>& filters,
                                                         const Strides& strides,
                                                         const Strides& dilations,
                                                         const CoordinateDiff& output_padding,
                                                         const Shape& output_shape,
                                                         const std::size_t groups)
    : GroupConvolutionTranspose(data,
                                filters,
                                strides,
                                dilations,
                                CoordinateDiff(),
                                CoordinateDiff(),
                                output_padding,
                                groups,
                                PadType::EXPLICIT,
                                output_shape)
{
}

op::GroupConvolutionTranspose::GroupConvolutionTranspose(const std::shared_ptr<Node>& data,
                                                         const std::shared_ptr<Node>& filters,
                                                         const Shape& output_shape,
                                                         const std::size_t groups)
    : GroupConvolutionTranspose(data,
                                filters,
                                Strides(),
                                Strides(),
                                CoordinateDiff(),
                                CoordinateDiff(),
                                CoordinateDiff(),
                                groups,
                                PadType::EXPLICIT,
                                output_shape)
{
}

void op::GroupConvolutionTranspose::pre_validate_and_infer_types()
{
    auto data_pshape = get_input_partial_shape(0);
    auto filters_pshape = get_input_partial_shape(1);
    if (data_pshape.is_static() && filters_pshape.is_static())
    {
        const Shape& data_shape = data_pshape.to_shape();
        const Shape& filters_shape = filters_pshape.to_shape();
        size_t n_data_channels{data_shape.at(1)};
        size_t n_filters_channels{filters_shape.at(0)};

        // groups
        NODE_VALIDATION_CHECK(this,
                              (m_groups <= n_data_channels && m_groups <= n_filters_channels),
                              "Incorrect value of groups: ",
                              m_groups);
        // filter channels
        NODE_VALIDATION_CHECK(
            this,
            n_filters_channels == n_data_channels,
            "Number of filters channels must be equal to number of data channels.");
        // data channels
        NODE_VALIDATION_CHECK(this,
                              n_data_channels % m_groups == 0,
                              "Number of data channels not a multiple of group size.");
        // padding type
        NODE_VALIDATION_CHECK(
            this, m_pad_type == PadType::EXPLICIT, "Currently only eplicit pad type is supported.");

        if (m_padding_begin.size() == 0)
        {
            m_padding_begin = conv_default_padding(this, data_pshape, filters_pshape);
        }
        if (m_padding_end.size() == 0)
        {
            m_padding_end = conv_default_padding(this, data_pshape, filters_pshape);
        }
        if (m_output_padding.size() == 0)
        {
            m_output_padding = conv_default_padding(this, data_pshape, filters_pshape);
        }
        if (m_strides.size() == 0)
        {
            m_strides = conv_default_strides(this, data_pshape, filters_pshape);
        }
        if (m_dilations.size() == 0)
        {
            m_dilations = conv_default_strides(this, data_pshape, filters_pshape);
        }

        const size_t num_spatial_dims = data_shape.size() - 2;

        NODE_VALIDATION_CHECK(this,
                              m_strides.size() == num_spatial_dims,
                              "Strides should be of number of input data features size.");

        NODE_VALIDATION_CHECK(this,
                              m_dilations.size() == num_spatial_dims,
                              "Dilations should be of number of input data features size.");

        NODE_VALIDATION_CHECK(this,
                              m_output_padding.size() == num_spatial_dims,
                              "Output padding should be of number of input data features size.");

        // If output shape is provided, ignore current values for padding begin/end and infer them.
        if (!m_output_shape.empty())
        {
            m_padding_begin = CoordinateDiff(num_spatial_dims);
            m_padding_end = CoordinateDiff(num_spatial_dims);

            Shape out_shape(m_output_shape);

            if (out_shape.size() > num_spatial_dims)
            {
                out_shape.erase(std::begin(out_shape), std::begin(out_shape) + 2);
            }

            for (int i = 0; i < num_spatial_dims; ++i)
            {
                int total_padding = m_strides[i] * (data_shape[i + 2] - 1) +
                                    m_dilations[i] * (filters_shape[i + 2] - 1) - out_shape[i] +
                                    m_output_padding[i] + 1;
                m_padding_begin[i] = total_padding / 2;
            }
            m_padding_end = m_padding_begin;
        }
    }
}

shared_ptr<Node> op::GroupConvolutionTranspose::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::GroupConvolutionTranspose>(new_args.at(0),
                                                      new_args.at(1),
                                                      get_strides(),
                                                      get_dilations(),
                                                      get_padding_begin(),
                                                      get_padding_end(),
                                                      get_output_padding(),
                                                      get_groups(),
                                                      get_pad_type(),
                                                      get_output_shape());
}

Shape op::GroupConvolutionTranspose::get_data_batch_shape() const
{
    const auto& data_shape = get_argument(0)->get_shape();
    const auto& filters_shape = get_argument(1)->get_shape();
    const size_t num_spatial_dims = data_shape.size() - 2;

    Shape data_batch_shape(data_shape.size(), 1);
    data_batch_shape.at(0) = data_shape.at(0);
    data_batch_shape.at(1) = filters_shape.at(1);

    if (m_output_shape.empty())
    {
        for (size_t i = 0; i < num_spatial_dims; ++i)
        {
            data_batch_shape[i + 2] = m_strides[i] * (data_shape[i + 2] - 1) +
                                      m_dilations[i] * (filters_shape[i + 2] - 1) -
                                      m_padding_begin[i] - m_padding_end[i] + m_output_padding[i] +
                                      1;
        }
    }
    else
    {
        Shape output_shape(m_output_shape);
        if (output_shape.size() > num_spatial_dims)
        {
            output_shape.erase(std::begin(output_shape), std::begin(output_shape) + 2);
        }

        for (size_t i = 0; i < num_spatial_dims; ++i)
        {
            data_batch_shape[i + 2] = output_shape[i];
        }
    }
    return data_batch_shape;
}

NodeVector op::GroupConvolutionTranspose::decompose_op() const
{
    auto data = get_argument(0);
    auto filters = get_argument(1);

    const Shape data_batch_shape = get_data_batch_shape();
    const size_t num_spatial_dims = data->get_shape().size() - 2;

    if (m_groups > 1)
    {
        // Split one convolution op to N ops where N is the number of groups
        // and concat results after computation.
        const size_t n_data_channels{data->get_shape().at(1)};
        const size_t n_filters_channels{filters->get_shape().at(0)};
        const size_t data_group_size{n_data_channels / m_groups};
        const size_t filters_group_size{n_filters_channels / m_groups};
        NodeVector convolution_nodes;

        // initial bounds for slice
        vector<size_t> data_lower_bounds(data->get_shape().size());
        vector<size_t> data_upper_bounds{data->get_shape()};
        vector<size_t> filters_lower_bounds(filters->get_shape().size());
        vector<size_t> filters_upper_bounds{filters->get_shape()};

        for (size_t group{0}; group < m_groups; ++group)
        {
            // slice data
            data_lower_bounds[1] = group * data_group_size;
            data_upper_bounds[1] = (group + 1) * data_group_size;
            auto sliced_data = make_shared<op::Slice>(data, data_lower_bounds, data_upper_bounds);
            // slice filters
            filters_lower_bounds[0] = group * filters_group_size;
            filters_upper_bounds[0] = (group + 1) * filters_group_size;
            auto sliced_filters =
                make_shared<op::Slice>(filters, filters_lower_bounds, filters_upper_bounds);

            convolution_nodes.push_back(
                make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         sliced_filters,
                                                         sliced_data,
                                                         m_strides,
                                                         m_dilations,
                                                         m_padding_begin,
                                                         m_padding_end,
                                                         Strides(num_spatial_dims, 1)));
        }
        size_t concatenation_axis = 1;
        return {make_shared<op::Concat>(convolution_nodes, concatenation_axis)};
    }
    else
    {
        return {make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                         filters,
                                                         data,
                                                         m_strides,
                                                         m_dilations,
                                                         m_padding_begin,
                                                         m_padding_end,
                                                         Strides(num_spatial_dims, 1))};
    }
}

void op::GroupConvolutionTranspose::generate_adjoints(autodiff::Adjoints& adjoints,
                                                      const NodeVector& deltas)
{
    throw ngraph_error(
        "Generating adjoints is not yet implemented for GroupConvolutionTranspose node.");
}
