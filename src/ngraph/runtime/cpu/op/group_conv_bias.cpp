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

#include <numeric>

#include "group_conv.hpp"
#include "group_conv_bias.hpp"

//#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void op::util::validate_groupconvbias_shapes(const Shape& data_shape,
                                             const Shape& filters_shape,
                                             const Shape& bias_shape)
{
    cout << "** GroupConvolutionBias validate_groupconvbias_shapes called \n";
    if (bias_shape.size() != 1)
    {
        throw ngraph_error("GroupConvolutionBias bias is expected to be 1D, but has shape: " +
                           vector_to_string(bias_shape));
    }
    if (bias_shape[0] != filters_shape[0] && bias_shape[0] != filters_shape[0] * filters_shape[1])
    {
        cout << "bias shape: " << bias_shape << ", filter_shape: " << filters_shape << "\n";
        throw ngraph_error(
            "GroupConvolutionBias bias element size does not match number of filters. bias_size "
            "= " +
            std::to_string(bias_shape[0]) + ", num_filters = " + std::to_string(filters_shape[0]));
    }

    // Note: the subtle match here
    if (data_shape[1] != filters_shape[0] && data_shape[1] != filters_shape[0] * filters_shape[1])
    {
        cout << " GroupConvolutionBias: data_shape: " << data_shape
             << ", filter_shape: " << filters_shape << "\n";
        // CHECK: Should we support if data_shape[1] == filters_shape[0] * filters_shape[1] ??
        throw ngraph_error(
            "GroupConvolution+bias data and filter have different number of channels: "
            "data_channel=" +
            std::to_string(data_shape[1]) + ", filter_channel= " +
            std::to_string(filters_shape[0]));
    }
}

Shape op::GroupConvolutionBias::get_weights_dimensions()
{
    // reshape weights into 5d tensors that includes groups
    const size_t OC = 0;
    const size_t OC_IN_OUTPUT = 1;
    const size_t IC = 1;

    cout << "\t Node name : " << get_name() << ", input_shape: " << get_inputs().at(0).get_shape()
         << "\n";
    Shape weights_shape_groups{get_inputs().at(1).get_shape()};
    cout << "\tnum_groups: " << get_groups() << ", get_wts_dims: " << weights_shape_groups << "\n";

    // when called from convertLayout, m_groups is 0, don't know why?!
    // hack for now
    if (get_groups() == 0 || get_groups() > get_inputs().at(0).get_shape().at(1))
    {
        m_groups = get_inputs().at(0).get_shape().at(1);
    }
    // adjust output and channel given a number of groups

    weights_shape_groups.at(OC) = get_shape().at(OC_IN_OUTPUT) / get_groups();
    cout << "\tOC, get_wts_dims: " << get_shape().at(OC_IN_OUTPUT) / get_groups() << " \n";

    weights_shape_groups.at(IC) = get_inputs().at(0).get_shape().at(IC) / get_groups();
    cout << "\tIC, get_wts_dims: " << get_inputs().at(0).get_shape().at(IC) / get_groups() << " \n";
    cout << "\twights_shape_groups: " << weights_shape_groups << "\n";
    // push_front the number of groups
    weights_shape_groups.insert(weights_shape_groups.begin(), get_groups());
    cout << "\tweights shape: " << weights_shape_groups << "\n\t-------------\n";
    return weights_shape_groups;
}

op::GroupConvolutionBias::GroupConvolutionBias(const shared_ptr<op::GroupConvolution>& conv,
                                               const shared_ptr<Node>& bias,
                                               const size_t groups,
                                               bool with_relu,
                                               float alpha)
    : Op("GroupConvolutionBias",
         check_single_output_args({conv->get_argument(0), conv->get_argument(1), bias}))
    , m_window_movement_strides(conv->get_window_movement_strides())
    , m_window_dilation_strides(conv->get_window_dilation_strides())
    , m_padding_below(conv->get_padding_below())
    , m_padding_above(conv->get_padding_above())
    , m_data_dilation_strides(conv->get_data_dilation_strides())
    , m_with_relu(with_relu)
    , m_groups(groups)
    , m_alpha(alpha)
{
    constructor_validate_and_infer_types();

    if (conv->get_element_type() != bias->get_element_type())
    {
        throw ngraph_error("GroupConvolution's element type isn't equal to bias!");
    }

    util::validate_groupconvbias_shapes(
        conv->get_argument(0)->get_shape(), conv->get_argument(1)->get_shape(), bias->get_shape());

    set_output_type(0, conv->get_element_type(), conv->get_shape());
}

op::GroupConvolutionBias::GroupConvolutionBias(const shared_ptr<Node>& data_batch,
                                               const shared_ptr<Node>& filters,
                                               const shared_ptr<Node>& bias,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               const size_t groups,
                                               bool with_relu,
                                               float alpha)
    : Op("GroupConvolutionBias", check_single_output_args({data_batch, filters, bias}))
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_with_relu(with_relu)
    , m_groups(groups)
    , m_alpha(alpha)
{
    cout << "** GroupConvolutionBias ctor called \n";
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
        throw ngraph_error("GroupConvolutionBias data batch and filter element types do not match");
    }
    util::validate_groupconvbias_shapes(data_batch_shape, filters_shape, bias->get_shape());

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
    cout << "** GroupConvolutionBias ctor done ** \n";
}

shared_ptr<Node> op::GroupConvolutionBias::copy_with_new_args(const NodeVector& new_args) const
{
    cout << "** GroupConvolutionBias copy_with_new_args called \n";
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return shared_ptr<Node>(new GroupConvolutionBias(new_args.at(0),
                                                     new_args.at(1),
                                                     new_args.at(2),
                                                     get_window_movement_strides(),
                                                     get_window_dilation_strides(),
                                                     get_padding_below(),
                                                     get_padding_above(),
                                                     get_data_dilation_strides(),
                                                     get_groups(),
                                                     m_with_relu,
                                                     get_alpha()));
}

void op::GroupConvolutionBias::generate_adjoints(autodiff::Adjoints& adjoints,
                                                 const NodeVector& deltas)
{
    throw ngraph_error("GroupConvBias generate_adjoints not supported implemented");
}
