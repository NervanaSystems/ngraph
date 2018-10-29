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
                                               const std::shared_ptr<Node> scale)
    : Op("QuantizedConvolution", check_single_output_args({data_batch, filters, scale}))
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

    auto scale_const_op = std::static_pointer_cast<ngraph::op::Constant>(scale);
    auto scale_val = scale_const_op->get_vector<float>();

    this->m_scale = scale_val[0];

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
shared_ptr<Node> op::QuantizedConvolution::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
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
                                                     new_args.at(2)));
}
