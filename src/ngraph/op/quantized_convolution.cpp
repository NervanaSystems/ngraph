//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include "quantized_convolution.hpp"
#include <numeric>
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::QuantizedConvolution::QuantizedConvolution(const shared_ptr<Node>& data_batch,
                                               const shared_ptr<Node>& filters,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               const std::shared_ptr<Node>& input_scale,
                                               const std::shared_ptr<Node>& input_zero_point,
                                               const std::shared_ptr<Node>& filter_scale,
                                               const std::shared_ptr<Node>& filter_zero_point,
                                               const std::shared_ptr<Node>& output_scale,
                                               const std::shared_ptr<Node>& output_zero_point,
                                               const ngraph::element::Type& output_type)
    : Op("QuantizedConvolution",
         check_single_output_args({data_batch,
                                   filters,
                                   input_scale,
                                   input_zero_point,
                                   filter_scale,
                                   filter_zero_point,
                                   output_scale,
                                   output_zero_point}))
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_output_type(output_type)
{
    constructor_validate_and_infer_types();

    auto& data_batch_shape = data_batch->get_shape();
    auto& filters_shape = filters->get_shape();

    set_output_type(0,
                    output_type,
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

#if 0
void op::QuantizedConvolution::validate_and_infer_types()
{
    const Shape& data_batch_shape = get_input_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const Shape& filters_shape = get_input_shape(1);
    element::Type filters_et = get_input_element_type(1);

    if (m_data_dilation_strides.size() == 0)
    {
        m_data_dilation_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_window_movement_strides.size() == 0)
    {
        m_window_movement_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_window_dilation_strides.size() == 0)
    {
        m_window_dilation_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_padding_below.size() == 0)
    {
        m_padding_below = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_padding_above.size() == 0)
    {
        m_padding_above = conv_default_padding(this, data_batch_shape, filters_shape);
    }

       set_output_type(0,
                    m_output_type,
                    util::infer_convolution_output_shape(this,
                                                         data_batch_shape,
                                                         filters_shape,
                                                         m_window_movement_strides,
                                                         m_window_dilation_strides,
                                                         m_padding_below,
                                                         m_padding_above,
                                                         m_data_dilation_strides,
                                                         0, /* batch_axis_data,              */
                                                         1, /* input_channel_axis_data,      */
                                                         1, /* input_channel_axis_filters,   */
                                                         0, /* output_channel_axis_filters,  */
                                                         0, /* batch_axis_result,            */
                                                         1  /* output_channel_axis_result,   */
                                                         ));

}
#endif

shared_ptr<Node> op::QuantizedConvolution::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return shared_ptr<Node>(new QuantizedConvolution(new_args.at(0),
                                                     new_args.at(1),
                                                     get_window_movement_strides(),
                                                     get_window_dilation_strides(),
                                                     get_padding_below(),
                                                     get_padding_above(),
                                                     get_data_dilation_strides(),
                                                     new_args.at(2),
                                                     new_args.at(3),
                                                     new_args.at(4),
                                                     new_args.at(5),
                                                     new_args.at(6),
                                                     new_args.at(7),
                                                     m_output_type));
}
