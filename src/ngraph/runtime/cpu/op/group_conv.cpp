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

#include "group_conv.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::GroupConvolution::GroupConvolution(const shared_ptr<Node>& data_batch,
                                       const shared_ptr<Node>& filters,
                                       const Strides& window_movement_strides,
                                       const Strides& window_dilation_strides,
                                       const CoordinateDiff& padding_below,
                                       const CoordinateDiff& padding_above,
                                       const Strides& data_dilation_strides,
                                       size_t groups,
                                       const Shape& output_shape)
    : RequiresTensorViewArgs("GroupConvolution", {data_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_groups(groups)
{
    auto& data_batch_et = data_batch->get_element_type();
    auto& filters_et = filters->get_element_type();

    //
    // Make sure data batch and filter element types match.
    //
    if (data_batch_et != filters_et)
    {
        throw ngraph_error("Convolution data batch and filter element types do not match");
    }

    set_value_type_checked(data_batch_et, output_shape);
}

Shape op::GroupConvolution::get_weights_dimensions() const
{
    return get_weights_dimensions(get_inputs().at(0)->get_shape(), get_inputs().at(1)->get_shape(), get_groups());
}


Shape op::GroupConvolution::get_weights_dimensions(const Shape& data_shape, const Shape& filters_shape, size_t groups)
{
    //reshape weights into 5d tensors that includes groups
    const size_t OC = 0;
    const size_t IC = 1;
    Shape weights_shape_groups{filters_shape};
    //adjust output and channel given a number of groups

    weights_shape_groups.at(OC) /= groups;
    weights_shape_groups.at(IC) = data_shape.at(IC) / groups;
    //push_front the number of groups
    weights_shape_groups.insert(weights_shape_groups.begin(), groups);
    return weights_shape_groups;
}

shared_ptr<Node> op::GroupConvolution::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<op::GroupConvolution>(new_args.at(0),
                                             new_args.at(1),
                                             get_window_movement_strides(),
                                             get_window_dilation_strides(),
                                             get_padding_below(),
                                             get_padding_above(),
                                             get_data_dilation_strides(),
                                             get_groups(),
                                             this->get_shape());
}

void op::GroupConvolution::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("NYI");
}

op::GroupConvolutionBackpropData::GroupConvolutionBackpropData(const Shape& data_batch_shape,
                                                     size_t groups,
                                                     const shared_ptr<Node>& filters,
                                                     const shared_ptr<Node>& output_delta,
                                                     const Strides& window_movement_strides_forward,
                                                     const Strides& window_dilation_strides_forward,
                                                     const CoordinateDiff& padding_below_forward,
                                                     const CoordinateDiff& padding_above_forward,
                                                     const Strides& data_dilation_strides_forward)
    : RequiresTensorViewArgs("GroupConvolutionBackpropData", {filters, output_delta})
    , m_data_batch_shape(data_batch_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
    , m_groups(groups)
{
    auto& filters_shape = get_inputs().at(0).get_shape();
    auto& filters_et = get_inputs().at(0).get_element_type();
    auto& output_delta_shape = get_inputs().at(1).get_shape();
    auto& output_delta_et = get_inputs().at(1).get_element_type();

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

    set_value_type_checked(filters_et, data_batch_shape);
}

shared_ptr<Node> op::GroupConvolutionBackpropData::copy_with_new_args(const NodeVector& new_args) const
{
    throw ngraph_error("NYI");
}

op::GroupConvolutionBackpropFilters::ConvolutionBackpropFilters(
    const shared_ptr<Node>& data_batch,
    const Shape& filters_shape,
    size_t groups,
    const shared_ptr<Node>& output_delta,
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
    , m_groups(groups)
{
    auto& data_batch_shape = get_inputs().at(0).get_shape();
    auto& data_batch_et = get_inputs().at(0).get_element_type();
    auto& output_delta_shape = get_inputs().at(1).get_shape();
    auto& output_delta_et = get_inputs().at(1).get_element_type();

    //
    // Make sure data batch and output delta element types match.
    //
    if (data_batch_et != output_delta_et)
    {
        throw ngraph_error(
            "Convolution filter backprop data batch and output delta element types do not match");
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

    set_value_type_checked(data_batch_et, filters_shape);
}

shared_ptr<Node> op::GroupConvolutionBackpropFilters::copy_with_new_args(const NodeVector& new_args) const
{
    throw ngraph_error("NYI");
}
