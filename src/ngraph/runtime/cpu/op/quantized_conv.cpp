/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "quantized_conv.hpp"
#include <numeric>
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::QuantizedConvolution::QuantizedConvolution(const shared_ptr<Node>& data_batch,
                                               const shared_ptr<Node>& filters,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               const std::shared_ptr<Node> min_input,
                                               const std::shared_ptr<Node> max_input,
                                               const std::shared_ptr<Node> min_filter,
                                               const std::shared_ptr<Node> max_filter,
                                               const std::shared_ptr<Node> min_freezed_output,
                                               const std::shared_ptr<Node> max_freezed_output)
    : Op("QuantizedConvolution",
         check_single_output_args({data_batch,
                                   filters,
                                   min_input,
                                   max_input,
                                   min_filter,
                                   max_filter,
                                   min_freezed_output,
                                   max_freezed_output}))
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
{
    constructor_validate_and_infer_types();

    //TODO(nbpatel): Add checks.

    auto& data_batch_shape = data_batch->get_shape();
    auto& filters_shape = filters->get_shape();

    auto min_input_const_op = std::static_pointer_cast<ngraph::op::Constant>(min_input);
    auto max_input_const_op = std::static_pointer_cast<ngraph::op::Constant>(max_input);
    auto min_filter_const_op = std::static_pointer_cast<ngraph::op::Constant>(min_filter);
    auto max_filter_const_op = std::static_pointer_cast<ngraph::op::Constant>(max_filter);
    auto min_freezed_output_const_op =
        std::static_pointer_cast<ngraph::op::Constant>(min_freezed_output);
    auto max_freezed_output_const_op =
        std::static_pointer_cast<ngraph::op::Constant>(max_freezed_output);
    float input_min = *(static_cast<float const*>(min_input_const_op->get_data_ptr()));
    float input_max = *(static_cast<float const*>(max_input_const_op->get_data_ptr()));
    float filter_min = *(static_cast<float const*>(min_filter_const_op->get_data_ptr()));
    float filter_max = *(static_cast<float const*>(max_filter_const_op->get_data_ptr()));
    float output_min = *(static_cast<float const*>(min_freezed_output_const_op->get_data_ptr()));
    float output_max = *(static_cast<float const*>(max_freezed_output_const_op->get_data_ptr()));

    this->m_input_min = input_min;
    this->m_input_max = input_max;
    this->m_filter_min = filter_min;
    this->m_filter_max = filter_max;
    this->m_freezed_output_min = output_min;
    this->m_freezed_output_max = output_max;

    set_output_size(3);
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
    set_output_type(1, element::f32, Shape{1});
    set_output_type(2, element::f32, Shape{1});
}
shared_ptr<Node> op::QuantizedConvolution::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 8)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
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
                                                     new_args.at(7)));
}
