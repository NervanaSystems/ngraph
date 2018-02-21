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
    throw ngraph_error("Not implemented!");
}
