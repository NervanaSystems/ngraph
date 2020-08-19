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

#include "quantized_conv_bias.hpp"

#include "ngraph/op/convolution.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::QuantizedConvolutionBias::type_info;

op::v0::QuantizedConvolutionBias::QuantizedConvolutionBias(const Output<Node>& data_batch,
                                                           const Output<Node>& filters,
                                                           const Output<Node>& bias,
                                                           const Strides& window_movement_strides,
                                                           const Strides& window_dilation_strides,
                                                           const CoordinateDiff& padding_below,
                                                           const CoordinateDiff& padding_above,
                                                           const Strides& data_dilation_strides,
                                                           const Output<Node>& scale,
                                                           const bool with_relu)
    : Op({data_batch, filters, bias, scale})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();

    auto& data_batch_shape = data_batch.get_shape();
    auto& filters_shape = filters.get_shape();

    // TODO: call ngraph util
    // util::validate_convbias_shapes(data_batch_shape, filters_shape, bias->get_shape());

    auto output_et = with_relu ? element::u8 : element::i8;
    set_output_type(0,
                    output_et,
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

shared_ptr<Node>
    op::v0::QuantizedConvolutionBias::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 4)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<QuantizedConvolutionBias>(new_args.at(0),
                                                 new_args.at(1),
                                                 new_args.at(2),
                                                 get_window_movement_strides(),
                                                 get_window_dilation_strides(),
                                                 get_padding_below(),
                                                 get_padding_above(),
                                                 get_data_dilation_strides(),
                                                 new_args.at(3),
                                                 m_with_relu);
}

constexpr NodeTypeInfo op::v0::QuantizedConvolutionBiasAdd::type_info;

op::v0::QuantizedConvolutionBiasAdd::QuantizedConvolutionBiasAdd(
    const Output<Node>& data_batch,
    const Output<Node>& filters,
    const Output<Node>& bias,
    const Output<Node>& sum_input,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides,
    const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above,
    const Strides& data_dilation_strides,
    const Output<Node>& scale,
    const Output<Node>& sum_scale,
    const bool with_relu)
    : Op({data_batch, filters, bias, sum_input, scale, sum_scale})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();

    auto& data_batch_shape = data_batch.get_shape();
    auto& filters_shape = filters.get_shape();

    // TODO: call ngraph util
    // util::validate_convbias_shapes(data_batch_shape, filters_shape, bias->get_shape());

    auto output_et = with_relu ? element::u8 : element::i8;
    set_output_type(0,
                    output_et,
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

shared_ptr<Node>
    op::v0::QuantizedConvolutionBiasAdd::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 6)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<QuantizedConvolutionBiasAdd>(new_args.at(0),
                                                    new_args.at(1),
                                                    new_args.at(2),
                                                    new_args.at(3),
                                                    get_window_movement_strides(),
                                                    get_window_dilation_strides(),
                                                    get_padding_below(),
                                                    get_padding_above(),
                                                    get_data_dilation_strides(),
                                                    new_args.at(4),
                                                    new_args.at(5),
                                                    m_with_relu);
}

constexpr NodeTypeInfo op::v0::QuantizedConvolutionBiasSignedAdd::type_info;

op::v0::QuantizedConvolutionBiasSignedAdd::QuantizedConvolutionBiasSignedAdd(
    const Output<Node>& data_batch,
    const Output<Node>& filters,
    const Output<Node>& bias,
    const Output<Node>& sum_input,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides,
    const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above,
    const Strides& data_dilation_strides,
    const Output<Node>& scale,
    const Output<Node>& sum_scale,
    const bool with_relu)
    : Op({data_batch, filters, bias, sum_input, scale, sum_scale})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();

    auto& data_batch_shape = data_batch.get_shape();
    auto& filters_shape = filters.get_shape();

    // TODO: call ngraph util
    // util::validate_convbias_shapes(data_batch_shape, filters_shape, bias->get_shape());

    // TODO (nbpatel): Remove with_relu arg from the API
    NGRAPH_CHECK(with_relu == true, "with_relu must be true");
    set_output_type(0,
                    element::i8,
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

shared_ptr<Node> op::v0::QuantizedConvolutionBiasSignedAdd::clone_with_new_inputs(
    const OutputVector& new_args) const
{
    if (new_args.size() != 6)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<QuantizedConvolutionBiasSignedAdd>(new_args.at(0),
                                                          new_args.at(1),
                                                          new_args.at(2),
                                                          new_args.at(3),
                                                          get_window_movement_strides(),
                                                          get_window_dilation_strides(),
                                                          get_padding_below(),
                                                          get_padding_above(),
                                                          get_data_dilation_strides(),
                                                          new_args.at(4),
                                                          new_args.at(5),
                                                          m_with_relu);
}
