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

#include "quantized_conv.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::QuantizedConvolution::QuantizedConvolution(const shared_ptr<Node>& data_batch,
                                               const shared_ptr<Node>& filters,
                                               const shared_ptr<Node>& bias,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               const float min_input,
                                               const float max_input,
                                               const float min_filter,
                                               const float max_filter,
                                               const float min_output,
                                               const float max_output)
    : RequiresTensorViewArgs("QuantizedConvolution", {data_batch, filters, bias})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_min_input(min_input)
    , m_max_input(max_input)
    , m_min_filter(min_filter)
    , m_max_filter(max_filter)
    , m_min_output(min_output)
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
        throw ngraph_error("QuantizedConvolution data batch and filter element types do not match");
    }

    set_value_type_checked(
        data_batch_et,
        util::infer_convolution_output_shape(data_batch_shape,
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
                                             1, /* output_channel_axis_result,   */
                                             ""));
}
