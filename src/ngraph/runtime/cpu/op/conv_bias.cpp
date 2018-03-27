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

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;
static Shape infer_convolution_output_shape(const Shape& data_batch_shape,
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
                                            string error_prefix)
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
op::ConvolutionBias::ConvolutionBias(const shared_ptr<op::Convolution>& conv,
                                     const shared_ptr<Node>& bias)
    : RequiresTensorViewArgs("ConvolutionBias",
                             {conv->get_input_op(0), conv->get_input_op(1), bias})
    , m_window_movement_strides(conv->get_window_movement_strides())
    , m_window_dilation_strides(conv->get_window_dilation_strides())
    , m_padding_below(conv->get_padding_below())
    , m_padding_above(conv->get_padding_above())
    , m_data_dilation_strides(conv->get_data_dilation_strides())
{
    if (conv->get_element_type() != bias->get_element_type())
    {
        throw ngraph_error("Convolution's element type isn't equal to bias!");
    }

    set_value_type_checked(conv->get_element_type(), conv->get_shape());
}

op::ConvolutionBias::ConvolutionBias(const shared_ptr<Node>& data_batch,
                                     const shared_ptr<Node>& filters,
                                     const shared_ptr<Node>& bias,
                                     const Strides& window_movement_strides,
                                     const Strides& window_dilation_strides,
                                     const CoordinateDiff& padding_below,
                                     const CoordinateDiff& padding_above,
                                     const Strides& data_dilation_strides)
    : RequiresTensorViewArgs("ConvolutionBias", {data_batch, filters, bias})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
{

    auto& data_batch_shape = data_batch->get_shape();
    auto& data_batch_et = data_batch->get_element_type();
    auto& filters_shape = filters->get_shape();
    auto& filters_et = filters->get_element_type();

    //
    // Make sure data batch and filter element types match.
    //
    if (data_batch_et != filters_et)
    {
        throw ngraph_error("Convolution data batch and filter element types do not match");
    }

    set_value_type_checked(data_batch_et,
                           infer_convolution_output_shape(data_batch_shape,
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

shared_ptr<Node> op::ConvolutionBias::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<ConvolutionBias>(new_args.at(0),
                                        new_args.at(1),
                                        new_args.at(2),
                                        get_window_movement_strides(),
                                        get_window_dilation_strides(),
                                        get_padding_below(),
                                        get_padding_above(),
                                        get_data_dilation_strides());
}

void op::ConvolutionBias::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const shared_ptr<Node>& delta)
{
    auto data = get_input_op(0);
    const auto data_shape = data->get_shape();

    auto filter = get_input_op(1);
    const auto filter_shape = filter->get_shape();

    auto bias = get_input_op(2);
    const auto bias_shape = bias->get_shape();

    // using regular convolution backprop for data
    adjoints.add_delta(data,
                       make_shared<op::ConvolutionBackpropData>(data_shape,
                                                                filter,
                                                                delta,
                                                                m_window_movement_strides,
                                                                m_window_dilation_strides,
                                                                m_padding_below,
                                                                m_padding_above,
                                                                m_data_dilation_strides));

    auto filter_bias_backprop =
        make_shared<op::ConvolutionBiasBackpropFiltersBias>(data,
                                                            filter_shape,
                                                            bias_shape,
                                                            delta,
                                                            m_window_movement_strides,
                                                            m_window_dilation_strides,
                                                            m_padding_below,
                                                            m_padding_above,
                                                            m_data_dilation_strides);
    auto filter_delta = make_shared<op::GetOutputElement>(filter_bias_backprop, 0);
    auto bias_delta = make_shared<op::GetOutputElement>(filter_bias_backprop, 1);

    adjoints.add_delta(filter, filter_delta);
    adjoints.add_delta(bias, bias_delta);
}

op::ConvolutionBiasBackpropFiltersBias::ConvolutionBiasBackpropFiltersBias(
    const shared_ptr<Node>& data_batch,
    const Shape& filters_shape,
    const Shape& bias_shape,
    const shared_ptr<Node>& output_delta,
    const Strides& window_movement_strides_forward,
    const Strides& window_dilation_strides_forward,
    const CoordinateDiff& padding_below_forward,
    const CoordinateDiff& padding_above_forward,
    const Strides& data_dilation_strides_forward)
    : RequiresTensorViewArgs("ConvolutionBiasBackpropFiltersBias", {data_batch, output_delta})
    , m_filters_shape(filters_shape)
    , m_bias_shape(bias_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
    auto& data_batch_shape = get_input_shape(0);
    auto& data_batch_et = get_input_element_type(0);
    auto& output_delta_et = get_input_element_type(1);

    //
    // Make sure data batch and output delta element types match.
    //
    if (data_batch_et != output_delta_et)
    {
        throw ngraph_error(
            "ConvolutionBiasBackpropFilterBias data batch and output delta element types do not "
            "match");
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

    add_output(data_batch_et, filters_shape);
    add_output(data_batch_et, bias_shape);
}

shared_ptr<Node>
    op::ConvolutionBiasBackpropFiltersBias::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<ConvolutionBiasBackpropFiltersBias>(new_args.at(0),
                                                           m_filters_shape,
                                                           m_bias_shape,
                                                           new_args.at(1),
                                                           m_window_movement_strides_forward,
                                                           m_window_dilation_strides_forward,
                                                           m_padding_below_forward,
                                                           m_padding_above_forward,
                                                           m_data_dilation_strides_forward);
}
