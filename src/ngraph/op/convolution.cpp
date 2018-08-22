/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <numeric>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

Shape op::util::infer_convolution_output_shape(const Shape& data_batch_shape,
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
                                               size_t output_channel_axis_result,
                                               const string& error_prefix)
{
    if (batch_axis_data > 1 || input_channel_axis_data > 1 || input_channel_axis_filters > 1 ||
        output_channel_axis_filters > 1 || batch_axis_result > 1 || output_channel_axis_result > 1)
    {
        throw ngraph_error(
            error_prefix +
            "Internal nGraph error: infer_convolution_output_shape: batch_axis_data, "
            "input_channel_axis_data, input_channel_axis_filters, "
            "output_channel_axis_filters, "
            "batch_axis_result, and output_channel_axis_result must all be 0 or 1.");
    }

    //
    // Make sure data_batch: NCiDi for some Di of rank>0, N != 0, Ci != 0.
    //
    if (data_batch_shape.size() < 3)
    {
        throw ngraph_error(
            error_prefix +
            "Convolution data batch input must have rank of at least 3 (one batch axis, one "
            "input-channel axis, at least one spatial dimension).");
    }

    size_t batch_size = data_batch_shape[batch_axis_data];
    if (batch_size == 0)
    {
        throw ngraph_error(error_prefix + "Convolution data batch size is zero.");
    }

    size_t input_channel_count = data_batch_shape[input_channel_axis_data];
    if (input_channel_count == 0)
    {
        throw ngraph_error(error_prefix + "Convolution requires at least one input channel.");
    }

    size_t spatial_dimension_count = data_batch_shape.size() - 2;

    //
    // Make sure filters: CoCiWv for some Co>0, rank of W = rank of Di.
    //
    if (filters_shape.size() != 2 + spatial_dimension_count)
    {
        throw ngraph_error(error_prefix +
                           "Convolution filter input must have rank of 2 + n_spatial_dimensions.");
    }

    size_t output_channel_count = filters_shape[output_channel_axis_filters];
    if (output_channel_count == 0)
    {
        throw ngraph_error(error_prefix + "Convolution requires at least one output channel.");
    }

    if (filters_shape[input_channel_axis_filters] != input_channel_count)
    {
        throw ngraph_error(error_prefix +
                           "Convolution data batch and filter input channel counts do not match.");
    }

    //
    // Make sure window movement strides, window dilation strides, and data dilation strides
    // have same rank as Di.
    //
    if (window_movement_strides.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            error_prefix +
            "Convolution window movement stride rank does not match number of spatial dimensions.");
    }

    if (window_dilation_strides.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            error_prefix +
            "Convolution window dilation stride rank does not match number of spatial dimensions.");
    }

    if (data_dilation_strides.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            error_prefix +
            "Convolution data dilation stride rank does not match number of spatial dimensions.");
    }

    //
    // Make sure padding-below and padding-above shapes have same rank as Di.
    //
    if (padding_below.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            error_prefix +
            "Convolution padding-below rank does not match number of spatial dimensions.");
    }

    if (padding_above.size() != spatial_dimension_count)
    {
        throw ngraph_error(
            error_prefix +
            "Convolution padding-above rank does not match number of spatial dimensions.");
    }

    //
    // Extract input item shape Di and make sure all dimensions are larger than 0 after padding and dilation.
    //
    Shape input_item_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (data_dilation_strides[i] == 0)
        {
            throw ngraph_error(error_prefix + "Convolution data dilation stride is zero.");
        }

        size_t dim_size = data_batch_shape[1 + 1 + i];
        size_t dilated_dim_size = (dim_size - 1) * data_dilation_strides[i] + 1;

        ptrdiff_t padded_dilated_dim_size = padding_below[i] + dilated_dim_size + padding_above[i];

        if (padded_dilated_dim_size < 0)
        {
            throw ngraph_error(
                error_prefix +
                "Convolution input spatial dimension after padding and dilation is negative.");
        }

        input_item_virtual_shape.push_back(padded_dilated_dim_size);

        if (input_item_virtual_shape[i] == 0)
        {
            throw ngraph_error(
                error_prefix +
                "Convolution input spatial dimension after dilation is zero even with padding.");
        }
    }

    //
    // Extract the physical shape Wp of the convolution window, *not* including dilation, from the filter dimensions.
    // At the same time, make sure window shape dimensions are all larger than 0.
    //
    Shape window_physical_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        window_physical_shape.push_back(filters_shape[1 + 1 + i]);
        if (window_physical_shape[i] == 0)
        {
            throw ngraph_error(error_prefix + "Convolution window shape has a zero-length axis.");
        }
    }

    //
    // Compute physical shape Wp of the convolution window, *including* dilation. At the same time, make sure all
    // window dilation strides are larger than 0, and that the dilated filter fits within the spatial dimensions.
    //
    Shape window_virtual_shape;

    for (size_t i = 0; i < spatial_dimension_count; i++)
    {
        if (window_dilation_strides[i] == 0)
        {
            throw ngraph_error(error_prefix + "Convolution window axis dilation stride is zero.");
        }

        window_virtual_shape.push_back((window_physical_shape[i] - 1) * window_dilation_strides[i] +
                                       1);

        if (window_virtual_shape[i] > input_item_virtual_shape[i])
        {
            throw ngraph_error(error_prefix +
                               "Convolution window after dilation is larger than the spatial "
                               "dimensions even with padding.");
        }
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
        if (window_movement_strides[i] == 0)
        {
            throw ngraph_error(error_prefix + "Convolution window axis movement stride is zero.");
        }
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
    : RequiresTensorViewArgs("Convolution", {data_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
{
    auto& data_batch_shape = get_inputs().at(0).get_shape();
    auto& data_batch_et = get_inputs().at(0).get_element_type();
    auto& filters_shape = get_inputs().at(1).get_shape();
    auto& filters_et = get_inputs().at(1).get_element_type();

    //
    // Make sure data batch and filter element types match.
    //
    if (data_batch_et != filters_et)
    {
        throw ngraph_error("Convolution data batch and filter element types do not match");
    }

    set_value_type_checked(data_batch_et,
                           util::infer_convolution_output_shape(data_batch_shape,
                                                                filters_shape,
                                                                window_movement_strides,
                                                                window_dilation_strides,
                                                                padding_below,
                                                                padding_above,
                                                                data_dilation_strides,
                                                                0,
                                                                1,
                                                                1,
                                                                0,
                                                                0,
                                                                1,
                                                                ""));
}

Strides op::Convolution::default_strides(const shared_ptr<Node>& data_batch)
{
    auto& data_batch_shape = data_batch->get_shape();
    if (data_batch_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Convolution data batch input must have rank of at least 3 (one batch axis, one "
            "input-channel axis, at least one spatial dimension).");
    }
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
                  default_strides(data_batch))
{
}

CoordinateDiff op::Convolution::default_padding(const shared_ptr<Node>& data_batch)
{
    auto& data_batch_shape = data_batch->get_shape();
    if (data_batch_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Convolution data batch input must have rank of at least 3 (one batch axis, one "
            "input-channel axis, at least one spatial dimension).");
    }
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
                  default_padding(data_batch),
                  default_padding(data_batch))
{
}

op::Convolution::Convolution(const shared_ptr<Node>& data_batch,
                             const shared_ptr<Node>& filters,
                             const Strides& window_movement_strides)
    : Convolution(data_batch,
                  filters,
                  window_movement_strides,
                  default_strides(data_batch),
                  default_padding(data_batch),
                  default_padding(data_batch))
{
}

op::Convolution::Convolution(const shared_ptr<Node>& data_batch, const shared_ptr<Node>& filters)
    : Convolution(data_batch,
                  filters,
                  default_strides(data_batch),
                  default_strides(data_batch),
                  default_padding(data_batch),
                  default_padding(data_batch))
{
}

shared_ptr<Node> op::Convolution::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
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
    : RequiresTensorViewArgs("ConvolutionBackpropData", {filters, output_delta})
    , m_data_batch_shape(data_batch_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
    auto& filters_shape = get_inputs().at(0).get_shape();
    auto& filters_et = get_inputs().at(0).get_element_type();
    auto& output_delta_shape = get_inputs().at(1).get_shape();
    auto& output_delta_et = get_inputs().at(1).get_element_type();

    //
    // Make sure filter and output delta element types match.
    //
    if (filters_et != output_delta_et)
    {
        throw ngraph_error(
            "Convolution data batch backprop filter and output delta element types do not match");
    }

    //                              Forward               Backward
    // Window movement strides      q                     p_x
    // Window dilation strides      p_f                   p_f
    // Padding below                a_x                   (S_F - 1)p_f - a_x
    // Padding above                b_x                   (S_f - 1)p_f + ((a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f) % q) - b_x
    // Data dilation strides        p_x                   q

    for (size_t i = 0; i < data_batch_shape.size() - 2; i++)
    {
        m_window_movement_strides_backward.push_back(data_dilation_strides_forward[i]);
        m_window_dilation_strides_backward.push_back(window_dilation_strides_forward[i]);
        m_padding_below_backward.push_back((filters_shape[i + 2] - 1) *
                                               window_dilation_strides_forward[i] -
                                           padding_below_forward[i]);
        m_padding_above_backward.push_back(
            (filters_shape[i + 2] - 1) * window_dilation_strides_forward[i] +
            ((padding_below_forward[i] +
              (data_batch_shape[i + 2] - 1) * data_dilation_strides_forward[i] +
              padding_above_forward[i] -
              (filters_shape[i + 2] - 1) * window_dilation_strides_forward[i]) %
             window_movement_strides_forward[i]) -
            padding_above_forward[i]);
        m_data_dilation_strides_backward.push_back(window_movement_strides_forward[i]);
    }

    Shape inferred_convolution_output_shape =
        util::infer_convolution_output_shape(output_delta_shape,
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
                                             1,
                                             "In ConvolutionBackpropData: ");

    // Not sure if this can ever actually happen (i.e., I think it will trip on something else
    // inside infer_convolution_output_shape before we get here) but it seems worth checking.
    if (inferred_convolution_output_shape != data_batch_shape)
    {
        throw ngraph_error(
            "Convolution data batch backprop inferred output shape does not match "
            "specified data batch shape");
    }

    set_value_type_checked(filters_et, inferred_convolution_output_shape);
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
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
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
    : RequiresTensorViewArgs("ConvolutionBackpropFilters", {data_batch, output_delta})
    , m_filters_shape(filters_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
    auto& data_batch_shape = get_inputs().at(0).get_shape();
    auto& data_batch_et = get_inputs().at(0).get_element_type();
    auto& output_delta_shape = get_inputs().at(1).get_shape();
    auto& output_delta_et = get_inputs().at(1).get_element_type();

    //
    // Make sure data batch and output delta element types match.
    //
    if (data_batch_et != output_delta_et)
    {
        throw ngraph_error(
            "Convolution filter backprop data batch and output delta element types do not match");
    }

    //                              Forward               Backward
    // Window movement strides      q                     p_f
    // Window dilation strides      p_f                   q
    // Padding below                a_x                   a_x
    // Padding above                b_x                   b_x - (a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f) % q
    // Data dilation strides        p_x                   p_x

    for (size_t i = 0; i < filters_shape.size() - 2; i++)
    {
        m_window_movement_strides_backward.push_back(window_dilation_strides_forward[i]);
        m_window_dilation_strides_backward.push_back(window_movement_strides_forward[i]);
        m_padding_below_backward.push_back(padding_below_forward[i]);
        m_padding_above_backward.push_back(
            padding_above_forward[i] -
            (padding_below_forward[i] +
             (data_batch_shape[i + 2] - 1) * data_dilation_strides_forward[i] +
             padding_above_forward[i] -
             (filters_shape[i + 2] - 1) * window_dilation_strides_forward[i]) %
                window_movement_strides_forward[i]);
        m_data_dilation_strides_backward.push_back(data_dilation_strides_forward[i]);
    }

    Shape inferred_convolution_output_shape =
        util::infer_convolution_output_shape(data_batch_shape,
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
                                             0,
                                             "In ConvolutionBackpropFilters: ");

    // Not sure if this can ever actually happen (i.e., I think it will trip on something else
    // inside infer_convolution_output_shape before we get here) but it seems worth checking.
    if (inferred_convolution_output_shape != filters_shape)
    {
        throw ngraph_error(
            "Convolution filter backprop inferred output shape does not match "
            "specified filter shape");
    }

    set_value_type_checked(data_batch_et, inferred_convolution_output_shape);
}

shared_ptr<Node>
    op::ConvolutionBackpropFilters::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<ConvolutionBackpropFilters>(new_args.at(0),
                                                   m_filters_shape,
                                                   new_args.at(1),
                                                   m_window_movement_strides_forward,
                                                   m_window_dilation_strides_forward,
                                                   m_padding_below_forward,
                                                   m_padding_above_forward,
                                                   m_data_dilation_strides_forward);
}
