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

#include <numeric>

#include "conv_bias.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void op::util::validate_convbias_shapes(const Shape& data_shape,
                                        const Shape& filters_shape,
                                        const Shape& bias_shape)
{
    if (bias_shape.size() != 1)
    {
        throw ngraph_error("Convolution+bias bias is expected to be 1D, but has shape: " +
                           vector_to_string(bias_shape));
    }
    if (bias_shape[0] != filters_shape[0])
    {
        throw ngraph_error(
            "Convolution+bias bias element size does not match number of filters. bias_size = " +
            std::to_string(bias_shape[0]) + ", num_filters = " + std::to_string(filters_shape[0]));
    }
    if (data_shape[1] != filters_shape[1])
    {
        throw ngraph_error(
            "Convolution+bias data and filter have different number of channels: data_channel=" +
            std::to_string(data_shape[1]) + ", filter_channel= " +
            std::to_string(filters_shape[1]));
    }
}

op::ConvolutionBias::ConvolutionBias(const shared_ptr<op::Convolution>& conv,
                                     const shared_ptr<Node>& bias,
                                     const bool with_relu)
    : Op("ConvolutionBias",
         check_single_output_args({conv->get_argument(0), conv->get_argument(1), bias}))
    , m_window_movement_strides(conv->get_window_movement_strides())
    , m_window_dilation_strides(conv->get_window_dilation_strides())
    , m_padding_below(conv->get_padding_below())
    , m_padding_above(conv->get_padding_above())
    , m_data_dilation_strides(conv->get_data_dilation_strides())
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();

    if (conv->get_element_type() != bias->get_element_type())
    {
        throw ngraph_error("Convolution's element type isn't equal to bias!");
    }

    util::validate_convbias_shapes(
        conv->get_argument(0)->get_shape(), conv->get_argument(1)->get_shape(), bias->get_shape());

    set_output_type(0, conv->get_element_type(), conv->get_shape());
}

op::ConvolutionBias::ConvolutionBias(const shared_ptr<Node>& data_batch,
                                     const shared_ptr<Node>& filters,
                                     const shared_ptr<Node>& bias,
                                     const Strides& window_movement_strides,
                                     const Strides& window_dilation_strides,
                                     const CoordinateDiff& padding_below,
                                     const CoordinateDiff& padding_above,
                                     const Strides& data_dilation_strides,
                                     const bool with_relu)
    : Op("ConvolutionBias", check_single_output_args({data_batch, filters, bias}))
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();

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
    util::validate_convbias_shapes(data_batch_shape, filters_shape, bias->get_shape());

    set_output_type(0,
                    data_batch_et,
                    util::infer_convolution_output_shape(this,
                                                         data_batch_shape,
                                                         filters_shape,
                                                         window_movement_strides,
                                                         window_dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilation_strides,
                                                         0, /* batch_axis_data,              */
                                                         1, /* input_channel_axis_data,      */
                                                         1, /* input_channel_axis_filters,   */
                                                         0, /* output_channel_axis_filters,  */
                                                         0, /* batch_axis_result,            */
                                                         1  /* output_channel_axis_result,   */
                                                         ));
}

shared_ptr<Node> op::ConvolutionBias::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return shared_ptr<Node>(new ConvolutionBias(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                get_window_movement_strides(),
                                                get_window_dilation_strides(),
                                                get_padding_below(),
                                                get_padding_above(),
                                                get_data_dilation_strides(),
                                                m_with_relu));
}

void op::ConvolutionBias::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto data = get_argument(0);
    const auto data_shape = data->get_shape();

    auto filter = get_argument(1);
    const auto filter_shape = filter->get_shape();

    auto bias = get_argument(2);
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
    : Op("ConvolutionBiasBackpropFiltersBias", check_single_output_args({data_batch, output_delta}))
    , m_filters_shape(filters_shape)
    , m_bias_shape(bias_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
{
    constructor_validate_and_infer_types();

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

    util::validate_convbias_shapes(data_batch_shape, filters_shape, bias_shape);
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

    set_output_size(2);
    set_output_type(0, data_batch_et, filters_shape);
    set_output_type(1, data_batch_et, bias_shape);
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

op::ConvolutionBiasAdd::ConvolutionBiasAdd(const std::shared_ptr<op::ConvolutionBias>& conv,
                                           const std::shared_ptr<Node>& sum_input,
                                           bool with_relu)
    : Op("ConvolutionBiasAdd",
         check_single_output_args(
             {conv->get_argument(0), conv->get_argument(1), conv->get_argument(2), sum_input}))
    , m_window_movement_strides(conv->get_window_movement_strides())
    , m_window_dilation_strides(conv->get_window_dilation_strides())
    , m_padding_below(conv->get_padding_below())
    , m_padding_above(conv->get_padding_above())
    , m_data_dilation_strides(conv->get_data_dilation_strides())
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();

    util::validate_convbias_shapes(conv->get_argument(0)->get_shape(),
                                   conv->get_argument(1)->get_shape(),
                                   conv->get_argument(2)->get_shape());
    set_output_type(0, conv->get_element_type(), conv->get_shape());
}

op::ConvolutionBiasAdd::ConvolutionBiasAdd(const std::shared_ptr<Node>& data_batch,
                                           const std::shared_ptr<Node>& filters,
                                           const std::shared_ptr<Node>& bias,
                                           const std::shared_ptr<Node>& sum_input,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           bool with_relu)
    : Op("ConvolutionBiasAdd", check_single_output_args({data_batch, filters, bias, sum_input}))
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();

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

    util::validate_convbias_shapes(data_batch_shape, filters_shape, bias->get_shape());
    set_output_type(0,
                    data_batch_et,
                    util::infer_convolution_output_shape(this,
                                                         data_batch_shape,
                                                         filters_shape,
                                                         window_movement_strides,
                                                         window_dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilation_strides,
                                                         0, /* batch_axis_data,              */
                                                         1, /* input_channel_axis_data,      */
                                                         1, /* input_channel_axis_filters,   */
                                                         0, /* output_channel_axis_filters,  */
                                                         0, /* batch_axis_result,            */
                                                         1  /* output_channel_axis_result,   */
                                                         ));
}

std::shared_ptr<Node> op::ConvolutionBiasAdd::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 4)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return std::shared_ptr<Node>(new ConvolutionBiasAdd(new_args.at(0),
                                                        new_args.at(1),
                                                        new_args.at(2),
                                                        new_args.at(3),
                                                        get_window_movement_strides(),
                                                        get_window_dilation_strides(),
                                                        get_padding_below(),
                                                        get_padding_above(),
                                                        get_data_dilation_strides(),
                                                        m_with_relu));
}
