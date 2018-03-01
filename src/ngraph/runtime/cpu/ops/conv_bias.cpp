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

#include "ngraph/runtime/cpu/ops/conv_bias.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::ConvolutionBias::ConvolutionBias(const std::shared_ptr<op::Convolution>& conv,
                                     const std::shared_ptr<Node>& bias)
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

op::ConvolutionBias::ConvolutionBias(const std::shared_ptr<Node>& data_batch,
                                     const std::shared_ptr<Node>& filters,
                                     const std::shared_ptr<Node>& bias,
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
}

std::shared_ptr<Node> op::ConvolutionBias::copy_with_new_args(
    const std::vector<std::shared_ptr<Node>>& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return std::shared_ptr<Node>(new ConvolutionBias(new_args.at(0),
                                                     new_args.at(1),
                                                     new_args.at(2),
                                                     get_window_movement_strides(),
                                                     get_window_dilation_strides(),
                                                     get_padding_below(),
                                                     get_padding_above(),
                                                     get_data_dilation_strides()));
}

void op::ConvolutionBias::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const std::shared_ptr<Node>& delta)
{
    auto data = get_input_op(0);
    const auto data_shape = data->get_shape();

    auto filter = get_input_op(1);
    const auto filter_shape = filter->get_shape();

    auto bias = get_input_op(2);
    const auto bias_shape = bias->get_shape();

    adjoints.add_delta(data,
                       std::make_shared<op::ConvolutionBiasBackpropData>(data_shape,
                                                                     filter,
                                                                     delta,
                                                                     m_window_movement_strides,
                                                                     m_window_dilation_strides,
                                                                     m_padding_below,
                                                                     m_padding_above,
                                                                     m_data_dilation_strides));

    auto filter_bias_backprop = std::make_shared<op::ConvolutionBiasBackpropFiltersBias>(data,
                                                                        filter_shape,
                                                                        bias_shape,
                                                                        delta,
                                                                        m_window_movement_strides,
                                                                        m_window_dilation_strides,
                                                                        m_padding_below,
                                                                        m_padding_above,
                                                                        m_data_dilation_strides);
    auto filter_delta = std::make_shared<op::GetOutputElement>(filter_bias_backprop, 0);
    auto bias_delta = std::make_shared<op::GetOutputElement>(filter_bias_backprop, 1);

    adjoints.add_delta(filter, filter_delta);
    adjoints.add_delta(bias, bias_delta);
}


op::ConvolutionBiasBackpropData::ConvolutionBiasBackpropData(const Shape& data_batch_shape,
                                                         const std::shared_ptr<Node>& filters,
                                                         const std::shared_ptr<Node>& output_delta,
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
    auto& filters_shape = get_input_shape(0);
    auto& filters_et = get_input_element_type(0);
    auto& output_delta_shape = get_input_shape(0);
    auto& output_delta_et = get_input_element_type(1);

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

//    Shape inferred_convolution_output_shape =
//            infer_convolution_output_shape(output_delta_shape,
//                                           filters_shape,
//                                           m_window_movement_strides_backward,
//                                           m_window_dilation_strides_backward,
//                                           m_padding_below_backward,
//                                           m_padding_above_backward,
//                                           m_data_dilation_strides_backward,
//                                           0,
//                                           1,
//                                           0,
//                                           1,
//                                           0,
//                                           1,
//                                           "In ConvolutionBiasBackpropData: ");
//
//    // Not sure if this can ever actually happen (i.e., I think it will trip on something else
//    // inside infer_convolution_output_shape before we get here) but it seems worth checking.
//    if (inferred_convolution_output_shape != data_batch_shape)
//    {
//        throw ngraph_error(
//                "Convolution data batch backprop inferred output shape does not match "
//                        "specified data batch shape");
//    }

    set_value_type_checked(filters_et, data_batch_shape);
}

std::shared_ptr<Node> op::ConvolutionBiasBackpropData::copy_with_new_args(
        const std::vector<std::shared_ptr<Node>>& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<ConvolutionBiasBackpropData>(m_data_batch_shape,
                                                         new_args.at(0),
                                                         new_args.at(1),
                                                         m_window_movement_strides_forward,
                                                         m_window_dilation_strides_forward,
                                                         m_padding_below_forward,
                                                         m_padding_above_forward,
                                                         m_data_dilation_strides_forward);
}

op::ConvolutionBiasBackpropFiltersBias::ConvolutionBiasBackpropFiltersBias(
        const std::shared_ptr<Node>& data_batch,
        const Shape& filters_shape,
        const Shape& bias_shape,
        const std::shared_ptr<Node>& output_delta,
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
    auto& data_batch_shape = get_input_shape(0);
    auto& data_batch_et = get_input_element_type(0);
    auto& output_delta_shape = get_input_shape(1);
    auto& output_delta_et = get_input_element_type(1);

    //
    // Make sure data batch and output delta element types match.
    //
    if (data_batch_et != output_delta_et)
    {
        throw ngraph_error(
                "ConvolutionBias filter backprop data batch and output delta element types do not match");
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

//    Shape inferred_convolution_filters_shape =
//            infer_convolution_output_shape(data_batch_shape,
//                                           filters_delta_shape,
//                                           m_window_movement_strides_backward,
//                                           m_window_dilation_strides_backward,
//                                           m_padding_below_backward,
//                                           m_padding_above_backward,
//                                           m_data_dilation_strides_backward,
//                                           1,
//                                           0,
//                                           0,
//                                           1,
//                                           1,
//                                           0,
//                                           "In ConvolutionBiasBackpropFiltersBias: ");
//
//    // Not sure if this can ever actually happen (i.e., I think it will trip on something else
//    // inside infer_convolution_output_shape before we get here) but it seems worth checking.
//    if (inferred_convolution_filters_shape != filters_shape)
//    {
//        throw ngraph_error(
//                "ConvolutionBias filter bias backprop inferred output shape does not match "
//                        "specified filter shape");
//    }

    add_output(data_batch_et, filters_shape);
    add_output(data_batch_et, bias_shape);
}

std::shared_ptr<Node> op::ConvolutionBiasBackpropFiltersBias::copy_with_new_args(
        const std::vector<std::shared_ptr<Node>>& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<ConvolutionBiasBackpropFiltersBias>(new_args.at(0),
                                                                m_filters_shape,
                                                                m_bias_shape,
                                                                new_args.at(1),
                                                                m_window_movement_strides_forward,
                                                                m_window_dilation_strides_forward,
                                                                m_padding_below_forward,
                                                                m_padding_above_forward,
                                                                m_data_dilation_strides_forward);
}
