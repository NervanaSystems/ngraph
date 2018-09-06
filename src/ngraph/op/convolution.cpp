//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <numeric>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

Shape op::util::infer_convolution_output_shape(const Node* node,
                                               const Shape& data_batch_shape,
                                               const Shape& filters_shape,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               size_t batch_axis_data,
                                               size_t input_channel_axis_data,
                                               size_t input_channel_axis_filters,
                                               size_t output_channel_axis_filters,
                                               size_t batch_axis_result,
                                               size_t output_channel_axis_result)
{
    NODE_VALIDATION_ASSERT(node, batch_axis_data <= 1) << "(This is an internal nGraph error)";
    NODE_VALIDATION_ASSERT(node, input_channel_axis_data <= 1)
        << "(This is an internal nGraph error)";
    NODE_VALIDATION_ASSERT(node, input_channel_axis_filters <= 1)
        << "(This is an internal nGraph error)";
    NODE_VALIDATION_ASSERT(node, output_channel_axis_filters <= 1)
        << "(This is an internal nGraph error)";
    NODE_VALIDATION_ASSERT(node, batch_axis_result <= 1) << "(This is an internal nGraph error)";
    NODE_VALIDATION_ASSERT(node, output_channel_axis_result <= 1)
        << "(This is an internal nGraph error)";

    //
    // Make sure data_batch: NCiDi for some Di of rank>0, N != 0, Ci != 0.
    //
    NODE_VALIDATION_ASSERT(node, data_batch_shape.size() >= 3)
        << "Data batch input must have rank of at least 3 (one batch axis, "
        << "one input-channel axis, and at least one spatial dimension) "
        << "(data batch shape: " << data_batch_shape << ").";

    size_t batch_size = data_batch_shape[batch_axis_data];
    NODE_VALIDATION_ASSERT(node, batch_size != 0)
        << "Data batch size is zero (data batch shape: " << data_batch_shape << ", "
        << "batch axis is axis " << batch_axis_data << ").";

    size_t input_channel_count = data_batch_shape[input_channel_axis_data];
    NODE_VALIDATION_ASSERT(node, input_channel_count != 0)
        << "Input channel count is zero (data batch shape: " << data_batch_shape << ", "
        << "channel axis is axis " << input_channel_axis_data << ").";

    size_t spatial_dimension_count = data_batch_shape.size() - 2;

    //
    // Make sure filters: CoCiWv for some Co>0, rank of W = rank of Di.
    //
    NODE_VALIDATION_ASSERT(node, filters_shape.size() == 2 + spatial_dimension_count)
        << "Filter input must have rank equal to the data batch (one axis for output "
        << "channels, one axis for input channels, and the same number of spatial "
        << "dimensions as the data batch (filter input shape: " << filters_shape << ", "
        << "data batch shape: " << data_batch_shape << ").";

    size_t output_channel_count = filters_shape[output_channel_axis_filters];
    NODE_VALIDATION_ASSERT(node, output_channel_count != 0)
        << "Output channel count for filters is zero (filters shape: " << filters_shape << ", "
        << "output channels on axis " << output_channel_axis_filters << ").";

    NODE_VALIDATION_ASSERT(node, filters_shape[input_channel_axis_filters] == input_channel_count)
        << "Input channel count for filters (" << filters_shape[input_channel_axis_filters] << ") "
        << "does not match the number of channels in the data batch (" << input_channel_count
        << ") "
        << "(filter input shape: " << filters_shape << ", filter input channels on axis "
        << input_channel_axis_filters << "; data batch shape: " << data_batch_shape
        << ", data batch channels on axis " << batch_axis_data << ").";

    //
    // Make sure window movement strides, window dilation strides, and data dilation strides
    // have same rank as Di.
    //
    NODE_VALIDATION_ASSERT(node, window_movement_strides.size() == spatial_dimension_count)
        << "Rank of window movement strides does not match the number of spatial dimensions ("
        << spatial_dimension_count
        << ") in the data batch (window movement strides: " << window_movement_strides
        << ", data batch shape: " << data_batch_shape << ").";

    NODE_VALIDATION_ASSERT(node, window_dilation_strides.size() == spatial_dimension_count)
        << "Rank of window dilation strides does not match the number of spatial dimensions ("
        << spatial_dimension_count
        << ") in the data batch (window dilation strides: " << window_dilation_strides
        << ", data batch shape: " << data_batch_shape << ").";

    NODE_VALIDATION_ASSERT(node, data_dilation_strides.size() == spatial_dimension_count)
        << "Rank of data dilation strides does not match the number of spatial dimensions ("
        << spatial_dimension_count
        << ") in the data batch (data dilation strides: " << data_dilation_strides
        << ", data batch shape: " << data_batch_shape << ").";

    //
    // Make sure padding-below and padding-above shapes have same rank as Di.
    //
    NODE_VALIDATION_ASSERT(node, padding_below.size() == spatial_dimension_count)
        << "Rank of the padding below does not match the number of spatial dimensions ("
        << spatial_dimension_count << ") in the data batch (padding below: " << padding_below
        << ", data batch shape: " << data_batch_shape << ").";

    NODE_VALIDATION_ASSERT(node, padding_above.size() == spatial_dimension_count)
        << "Rank of the padding above does not match the number of spatial dimensions ("
        << spatial_dimension_count << ") in the data batch (padding above: " << padding_above
        << ", data batch shape: " << data_batch_shape << ").";

    //
    // Extract input item shape Di and make sure all dimensions are larger than 0 after padding and dilation.
    //
    std::vector<ptrdiff_t> input_item_virtual_shape_signed;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(node, data_dilation_strides[i] != 0)
            << "Data dilation stride at spatial dimension " << i << " is zero "
            << "(data dilation strides: " << data_dilation_strides << ").";

        size_t dim_size = data_batch_shape[1 + 1 + i];
        size_t dilated_dim_size = (dim_size - 1) * data_dilation_strides[i] + 1;

        ptrdiff_t padded_dilated_dim_size = padding_below[i] + dilated_dim_size + padding_above[i];

        input_item_virtual_shape_signed.push_back(padded_dilated_dim_size);
    }

    Shape input_item_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(node, input_item_virtual_shape_signed[i] > 0)
            << "Input dimension after padding and dilation is non-positive "
            << "at spatial axis " << i
            << " (post-padding/dilation input item shape: " << input_item_virtual_shape
            << ", data batch shape: " << data_batch_shape
            << ", data dilation strides: " << data_dilation_strides
            << ", padding below: " << padding_below << ", padding above: " << padding_above << ").";

        input_item_virtual_shape.push_back(size_t(input_item_virtual_shape_signed[i]));
    }

    //
    // Extract the physical shape Wp of the convolution window, *not* including dilation, from the filter dimensions.
    // At the same time, make sure window shape dimensions are all larger than 0.
    //
    Shape window_physical_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        window_physical_shape.push_back(filters_shape[1 + 1 + i]);
        NODE_VALIDATION_ASSERT(node, window_physical_shape[i] != 0)
            << "Filters shape at spatial dimension " << i << " is zero "
            << "(filters shape: " << filters_shape << ").";
    }

    //
    // Compute virtual shape Wp of the convolution window, *including* dilation. At the same time, make sure all
    // window dilation strides are larger than 0, and that the dilated filter fits within the spatial dimensions.
    //
    Shape window_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(node, window_dilation_strides[i] != 0)
            << "Window dilation stride at spatial dimension " << i << " is zero "
            << "(window dilation strides: " << window_dilation_strides << ").";

        window_virtual_shape.push_back((window_physical_shape[i] - 1) * window_dilation_strides[i] +
                                       1);

        NODE_VALIDATION_ASSERT(node, window_virtual_shape[i] <= input_item_virtual_shape[i])
            << "Post-dilation window shape is smaller than the post-padding/dilation "
            << "input item shape at spatial dimension " << i << " (post-padding/dilation "
            << "input item shape: " << input_item_virtual_shape
            << ", data batch shape: " << data_batch_shape
            << ", data dilation strides: " << data_dilation_strides
            << ", padding below: " << padding_below << ", padding above: " << padding_above
            << ", post-dilation window shape: " << window_virtual_shape
            << ", filters shape: " << filters_shape
            << ", window dilation strides: " << window_dilation_strides;
    }

    //
    // Construct result shape: NCoDo or CoNDo (depending on *_axis_result), checking at the same
    // time that all window movement strides are larger than 0.
    //
    Shape result_shape(spatial_dimension_count + 2);
    result_shape[batch_axis_result] = batch_size;
    result_shape[output_channel_axis_result] = output_channel_count;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        NODE_VALIDATION_ASSERT(node, window_movement_strides[i] != 0)
            << "Window movement stride at spatial dimension " << i << " is zero "
            << "(window movement strides: " << window_movement_strides << ").";

        result_shape[i + 2] = ceil_div(input_item_virtual_shape[i] - window_virtual_shape[i] + 1,
                                       window_movement_strides[i]);
    }

    return result_shape;
}

op::Convolution::Convolution(const shared_ptr<Node>& data_batch,
                             const shared_ptr<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides,
                             const CoordinateDiff& padding_below,
                             const CoordinateDiff& padding_above,
                             const Strides& data_dilation_strides)
    : Op("Convolution", check_single_output_args({data_batch, filters}))
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
{
    constructor_validate_and_infer_types();
}

void op::Convolution::validate_and_infer_types()
{
    auto& data_batch_shape = get_input_shape(0);
    auto& data_batch_et = get_input_element_type(0);
    auto& filters_shape = get_input_shape(1);
    auto& filters_et = get_input_element_type(1);

    if (m_data_dilation_strides.size() == 0)
    {
        m_data_dilation_strides = default_strides(this, data_batch_shape);
    }
    if (m_window_movement_strides.size() == 0)
    {
        m_window_movement_strides = default_strides(this, data_batch_shape);
    }
    if (m_window_dilation_strides.size() == 0)
    {
        m_window_dilation_strides = default_strides(this, data_batch_shape);
    }

    if (m_padding_below.size() == 0)
    {
        m_padding_below = default_padding(this, data_batch_shape);
    }

    if (m_padding_above.size() == 0)
    {
        m_padding_above = default_padding(this, data_batch_shape);
    }

    //
    // Make sure data batch and filter element types match.
    //
    NODE_VALIDATION_ASSERT(this, data_batch_et == filters_et)
        << "Element types for data batch and filters do not match (data batch element type: "
        << data_batch_et << ", filters element type: " << filters_et << ").";

    set_output_type(0,
                    data_batch_et,
                    util::infer_convolution_output_shape(this,
                                                         data_batch_shape,
                                                         filters_shape,
                                                         m_window_movement_strides,
                                                         m_window_dilation_strides,
                                                         m_padding_below,
                                                         m_padding_above,
                                                         m_data_dilation_strides,
                                                         0,
                                                         1,
                                                         1,
                                                         0,
                                                         0,
                                                         1));
}

Strides op::Convolution::default_strides(const Node* node, const Shape& data_batch_shape)
{
    // For consistency we should throw the same error message here that we throw in the constructor.
    NODE_VALIDATION_ASSERT(node, data_batch_shape.size() >= 3)
        << "Data batch input must have rank of at least 3 (one batch axis, "
        << "one input-channel axis, and at least one spatial dimension) "
        << "(data batch shape: " << data_batch_shape << ").";

    return Strides(data_batch_shape.size() - 2, 1);
}

op::Convolution::Convolution(const shared_ptr<Node>& data_batch,
                             const shared_ptr<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides,
                             const CoordinateDiff& padding_below,
                             const CoordinateDiff& padding_above)
    : Convolution(data_batch,
                  filters,
                  window_movement_strides,
                  window_dilation_strides,
                  padding_below,
                  padding_above,
                  Strides())
{
}

CoordinateDiff op::Convolution::default_padding(const Node* node, const Shape& data_batch_shape)
{
    // For consistency we should throw the same error message here that we throw in the constructor.
    NODE_VALIDATION_ASSERT(node, data_batch_shape.size() >= 3)
        << "Data batch input must have rank of at least 3 (one batch axis, "
        << "one input-channel axis, and at least one spatial dimension) "
        << "(data batch shape: " << data_batch_shape << ").";

    return CoordinateDiff(data_batch_shape.size() - 2, 0);
}

op::Convolution::Convolution(const shared_ptr<Node>& data_batch,
                             const shared_ptr<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides)
    : Convolution(data_batch,
                  filters,
                  window_movement_strides,
                  window_dilation_strides,
                  CoordinateDiff(),
                  CoordinateDiff())
{
}

op::Convolution::Convolution(const shared_ptr<Node>& data_batch,
                             const shared_ptr<Node>& filters,
                             const Strides& window_movement_strides)
    : Convolution(data_batch,
                  filters,
                  window_movement_strides,
                  Strides(),
                  CoordinateDiff(),
                  CoordinateDiff())
{
}

op::Convolution::Convolution(const shared_ptr<Node>& data_batch, const shared_ptr<Node>& filters)
    : Convolution(data_batch, filters, Strides(), Strides(), CoordinateDiff(), CoordinateDiff())
{
}

shared_ptr<Node> op::Convolution::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Convolution>(new_args.at(0),
                                    new_args.at(1),
                                    m_window_movement_strides,
                                    m_window_dilation_strides,
                                    m_padding_below,
                                    m_padding_above,
                                    m_data_dilation_strides);
}

void op::Convolution::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);
    const auto x_shape = x->get_shape();

    auto f = get_argument(1);
    const auto f_shape = f->get_shape();

    adjoints.add_delta(x,
                       make_shared<op::ConvolutionBackpropData>(x_shape,
                                                                f,
                                                                delta,
                                                                m_window_movement_strides,
                                                                m_window_dilation_strides,
                                                                m_padding_below,
                                                                m_padding_above,
                                                                m_data_dilation_strides));

    adjoints.add_delta(f,
                       make_shared<op::ConvolutionBackpropFilters>(x,
                                                                   f_shape,
                                                                   delta,
                                                                   m_window_movement_strides,
                                                                   m_window_dilation_strides,
                                                                   m_padding_below,
                                                                   m_padding_above,
                                                                   m_data_dilation_strides));
}

op::ConvolutionBackpropData::ConvolutionBackpropData(const Shape& data_batch_shape,
                                                     const shared_ptr<Node>& filters,
                                                     const shared_ptr<Node>& output_delta,
                                                     const Strides& window_movement_strides_forward,
                                                     const Strides& window_dilation_strides_forward,
                                                     const CoordinateDiff& padding_below_forward,
                                                     const CoordinateDiff& padding_above_forward,
                                                     const Strides& data_dilation_strides_forward)
    : Op("ConvolutionBackpropData", check_single_output_args({filters, output_delta}))
    , m_data_batch_shape(data_batch_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
    constructor_validate_and_infer_types();
}

void op::ConvolutionBackpropData::validate_and_infer_types()
{
    auto& filters_shape = get_input_shape(0);
    auto& filters_et = get_input_element_type(0);
    auto& output_delta_shape = get_input_shape(1);
    auto& output_delta_et = get_input_element_type(1);

    //
    // Make sure filter and output delta element types match.
    //
    NODE_VALIDATION_ASSERT(this, output_delta_et == filters_et)
        << "Element types for filters and output delta do not match (filters element type: "
        << filters_et << ", output delta element type: " << output_delta_et << ").";

    //                              Forward               Backward
    // Window movement strides      q                     p_x
    // Window dilation strides      p_f                   p_f
    // Padding below                a_x                   (S_F - 1)p_f - a_x
    // Padding above                b_x                   (S_f - 1)p_f + ((a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f) % q) - b_x
    // Data dilation strides        p_x                   q

    for (size_t i = 0; i < m_data_batch_shape.size() - 2; i++)
    {
        m_window_movement_strides_backward.push_back(m_data_dilation_strides_forward[i]);
        m_window_dilation_strides_backward.push_back(m_window_dilation_strides_forward[i]);
        m_padding_below_backward.push_back((filters_shape[i + 2] - 1) *
                                               m_window_dilation_strides_forward[i] -
                                           m_padding_below_forward[i]);
        m_padding_above_backward.push_back(
            (filters_shape[i + 2] - 1) * m_window_dilation_strides_forward[i] +
            ((m_padding_below_forward[i] +
              (m_data_batch_shape[i + 2] - 1) * m_data_dilation_strides_forward[i] +
              m_padding_above_forward[i] -
              (filters_shape[i + 2] - 1) * m_window_dilation_strides_forward[i]) %
             m_window_movement_strides_forward[i]) -
            m_padding_above_forward[i]);
        m_data_dilation_strides_backward.push_back(m_window_movement_strides_forward[i]);
    }

    Shape inferred_convolution_output_shape =
        util::infer_convolution_output_shape(this,
                                             output_delta_shape,
                                             filters_shape,
                                             m_window_movement_strides_backward,
                                             m_window_dilation_strides_backward,
                                             m_padding_below_backward,
                                             m_padding_above_backward,
                                             m_data_dilation_strides_backward,
                                             0,
                                             1,
                                             0,
                                             1,
                                             0,
                                             1);

    NODE_VALIDATION_ASSERT(this, inferred_convolution_output_shape == m_data_batch_shape)
        << "Specified data batch shape does not match the inferred data batch shape "
        << "(specified shape: " << m_data_batch_shape
        << ", inferred data batch shape: " << inferred_convolution_output_shape;

    set_output_type(0, filters_et, inferred_convolution_output_shape);
}

void op::ConvolutionBackpropData::generate_adjoints(autodiff::Adjoints& adjoints,
                                                    const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(1);
    const auto x_shape = x->get_shape();

    auto f = get_argument(0);
    const auto f_shape = f->get_shape();

    auto data_conv = make_shared<op::Convolution>(delta,
                                                  f,
                                                  m_window_movement_strides_forward,
                                                  m_window_dilation_strides_forward,
                                                  m_padding_below_forward,
                                                  m_padding_above_forward,
                                                  m_data_dilation_strides_forward);

    adjoints.add_delta(x, data_conv);

    Strides window_movement_strides;
    Strides window_dilation_strides;
    CoordinateDiff padding_below;
    CoordinateDiff padding_above;
    Strides data_dilation_strides;
    for (size_t i = 0; i < f_shape.size() - 2; i++)
    {
        window_movement_strides.push_back(m_window_dilation_strides_backward[i]);
        window_dilation_strides.push_back(m_window_movement_strides_backward[i]);
        padding_below.push_back(m_padding_below_backward[i]);
        padding_above.push_back(m_padding_above_backward[i] -
                                (m_padding_below_backward[i] +
                                 (x_shape[i + 2] - 1) * m_data_dilation_strides_backward[i] +
                                 m_padding_above_backward[i] -
                                 (f_shape[i + 2] - 1) * m_window_dilation_strides_backward[i]) %
                                    m_window_movement_strides_backward[i]);
        data_dilation_strides.push_back(m_data_dilation_strides_backward[i]);
    }

    auto swap_NC = [](const shared_ptr<Node> n) {
        AxisVector ax_order = ngraph::get_default_order(n->get_shape());
        ax_order[0] = 1;
        ax_order[1] = 0;

        auto new_shape = n->get_shape();
        new_shape[0] = n->get_shape()[1];
        new_shape[1] = n->get_shape()[0];

        return make_shared<op::Reshape>(n, ax_order, new_shape);
    };

    delta = swap_NC(delta);
    x = swap_NC(x);

    shared_ptr<Node> filter_deconv_bprop = make_shared<op::Convolution>(x,
                                                                        delta,
                                                                        window_movement_strides,
                                                                        window_dilation_strides,
                                                                        padding_below,
                                                                        padding_above,
                                                                        data_dilation_strides);
    AxisSet axes;
    for (size_t i = 2; i < filter_deconv_bprop->get_shape().size(); ++i)
    {
        axes.insert(i);
    }
    filter_deconv_bprop = make_shared<ngraph::op::Reverse>(filter_deconv_bprop, axes);
    adjoints.add_delta(f, filter_deconv_bprop);
}

shared_ptr<Node> op::ConvolutionBackpropData::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ConvolutionBackpropData>(m_data_batch_shape,
                                                new_args.at(0),
                                                new_args.at(1),
                                                m_window_movement_strides_forward,
                                                m_window_dilation_strides_forward,
                                                m_padding_below_forward,
                                                m_padding_above_forward,
                                                m_data_dilation_strides_forward);
}

op::ConvolutionBackpropFilters::ConvolutionBackpropFilters(
    const shared_ptr<Node>& data_batch,
    const Shape& filters_shape,
    const shared_ptr<Node>& output_delta,
    const Strides& window_movement_strides_forward,
    const Strides& window_dilation_strides_forward,
    const CoordinateDiff& padding_below_forward,
    const CoordinateDiff& padding_above_forward,
    const Strides& data_dilation_strides_forward)
    : Op("ConvolutionBackpropFilters", check_single_output_args({data_batch, output_delta}))
    , m_filters_shape(filters_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
    constructor_validate_and_infer_types();
}

void op::ConvolutionBackpropFilters::validate_and_infer_types()
{
    auto& data_batch_shape = get_input_shape(0);
    auto& data_batch_et = get_input_element_type(0);
    auto& output_delta_shape = get_input_shape(1);
    auto& output_delta_et = get_input_element_type(1);

    //
    // Make sure data batch and output delta element types match.
    //
    NODE_VALIDATION_ASSERT(this, output_delta_et == data_batch_et)
        << "Element types for data batch and output delta do not match (data batch element type: "
        << data_batch_et << ", output delta element type: " << output_delta_et << ").";

    //                              Forward               Backward
    // Window movement strides      q                     p_f
    // Window dilation strides      p_f                   q
    // Padding below                a_x                   a_x
    // Padding above                b_x                   b_x - (a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f) % q
    // Data dilation strides        p_x                   p_x

    for (size_t i = 0; i < m_filters_shape.size() - 2; i++)
    {
        m_window_movement_strides_backward.push_back(m_window_dilation_strides_forward[i]);
        m_window_dilation_strides_backward.push_back(m_window_movement_strides_forward[i]);
        m_padding_below_backward.push_back(m_padding_below_forward[i]);
        m_padding_above_backward.push_back(
            m_padding_above_forward[i] -
            (m_padding_below_forward[i] +
             (data_batch_shape[i + 2] - 1) * m_data_dilation_strides_forward[i] +
             m_padding_above_forward[i] -
             (m_filters_shape[i + 2] - 1) * m_window_dilation_strides_forward[i]) %
                m_window_movement_strides_forward[i]);
        m_data_dilation_strides_backward.push_back(m_data_dilation_strides_forward[i]);
    }

    Shape inferred_convolution_output_shape =
        util::infer_convolution_output_shape(this,
                                             data_batch_shape,
                                             output_delta_shape,
                                             m_window_movement_strides_backward,
                                             m_window_dilation_strides_backward,
                                             m_padding_below_backward,
                                             m_padding_above_backward,
                                             m_data_dilation_strides_backward,
                                             1,
                                             0,
                                             0,
                                             1,
                                             1,
                                             0);

    NODE_VALIDATION_ASSERT(this, inferred_convolution_output_shape == m_filters_shape)
        << "Specified filters shape does not match the inferred filters shape "
        << "(specified shape: " << m_filters_shape
        << ", inferred filters shape: " << inferred_convolution_output_shape;

    set_output_type(0, data_batch_et, inferred_convolution_output_shape);
}

shared_ptr<Node>
    op::ConvolutionBackpropFilters::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ConvolutionBackpropFilters>(new_args.at(0),
                                                   m_filters_shape,
                                                   new_args.at(1),
                                                   m_window_movement_strides_forward,
                                                   m_window_dilation_strides_forward,
                                                   m_padding_below_forward,
                                                   m_padding_above_forward,
                                                   m_data_dilation_strides_forward);
}
