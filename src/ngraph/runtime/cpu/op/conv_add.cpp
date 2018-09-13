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

#include "conv_add.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void op::util::validate_conv_shapes(const Node* node,
                                    const Shape& data_shape,
                                    const Shape& filters_shape)
{
    NODE_VALIDATION_ASSERT(node, data_shape[1] == filters_shape[1])
        << "Number of channels for data and filters do not match (data num channels: "
        << data_shape[1] << ", filters num channels: " << filters_shape[1] << ").";
}

op::ConvolutionAdd::ConvolutionAdd(const std::shared_ptr<op::Convolution>& conv,
                                   const std::shared_ptr<Node>& sum_input,
                                   bool with_relu)
    : Op("ConvolutionAdd",
         check_single_output_args({conv->get_argument(0), conv->get_argument(1), sum_input}))
    , m_window_movement_strides(conv->get_window_movement_strides())
    , m_window_dilation_strides(conv->get_window_dilation_strides())
    , m_padding_below(conv->get_padding_below())
    , m_padding_above(conv->get_padding_above())
    , m_data_dilation_strides(conv->get_data_dilation_strides())
    , m_with_relu(with_relu)
{
    constructor_validate_and_infer_types();
    util::validate_conv_shapes(
        this, conv->get_argument(0)->get_shape(), conv->get_argument(1)->get_shape());
    set_output_type(0, conv->get_element_type(), conv->get_shape());
}

op::ConvolutionAdd::ConvolutionAdd(const std::shared_ptr<Node>& data_batch,
                                   const std::shared_ptr<Node>& filters,
                                   const std::shared_ptr<Node>& sum_input,
                                   const Strides& window_movement_strides,
                                   const Strides& window_dilation_strides,
                                   const CoordinateDiff& padding_below,
                                   const CoordinateDiff& padding_above,
                                   const Strides& data_dilation_strides,
                                   bool with_relu)
    : Op("ConvolutionAdd", check_single_output_args({data_batch, filters, sum_input}))
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
    NODE_VALIDATION_ASSERT(this, data_batch_et == filters_et)
        << "Element types for data_batch and filters do not match (data batch element type: "
        << data_batch_et << ", filters element type: " << filters_et << ").";

    util::validate_conv_shapes(this, data_batch_shape, filters_shape);
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

std::shared_ptr<Node> op::ConvolutionAdd::copy_with_new_args(const NodeVector& new_args) const
{
    NODE_VALIDATION_ASSERT(this, new_args.size() == 3)
        << "New arg size is not 3 (new args size: " << new_args.size() << ").";

    return std::shared_ptr<Node>(new ConvolutionAdd(new_args.at(0),
                                                    new_args.at(1),
                                                    new_args.at(2),
                                                    get_window_movement_strides(),
                                                    get_window_dilation_strides(),
                                                    get_padding_below(),
                                                    get_padding_above(),
                                                    get_data_dilation_strides(),
                                                    m_with_relu));
}
