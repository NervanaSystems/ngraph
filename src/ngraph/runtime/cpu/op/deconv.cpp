//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "deconv.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::DeconvolutionBias::type_info;

op::DeconvolutionBias::DeconvolutionBias(const Shape& data_batch_shape,
                                         const Output<Node>& filters,
                                         const Output<Node>& output_delta,
                                         const Output<Node>& bias,
                                         const Strides& window_movement_strides_forward,
                                         const Strides& window_dilation_strides_forward,
                                         const CoordinateDiff& padding_below_forward,
                                         const CoordinateDiff& padding_above_forward,
                                         const Strides& data_dilation_strides_forward,
                                         const bool with_relu)
    : Op({filters, output_delta, bias})
    , m_data_batch_shape(data_batch_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
    , m_with_relu(with_relu)
{
    NGRAPH_DEBUG << "DeconvolutionBias ctor" << endl;
    NGRAPH_DEBUG << "data: " << data_batch_shape << ", filters: " << filters.get_shape()
                 << ", output_delta: " << output_delta.get_shape();
    constructor_validate_and_infer_types();
}

void op::DeconvolutionBias::validate_and_infer_types()
{
    NGRAPH_DEBUG << "DeconvolutionBias::validate_and_infer_types" << endl;
    // Backprop to data is itself convolution, with inputs/outputs/attributes transmogrified as
    // follows.
    //
    //                          Forward   Backward
    // "N" axis for data batch  0         0
    // "C" axis for data batch  1         1
    // "Co" axis for filters    0         0
    // "Ci" axis for filters    1         1
    // "N" axis for output      0         0
    // "C" axis for output      1         1
    // Data batch               x         delta
    // Data batch shape         S_x       S_o
    // Filters                  f         reverse(f) [on spatial axes]
    // Filters shape            S_f       S_f
    // Window movement strides  q_x       p_x
    // Window dilation strides  p_f       p_f
    // Padding below            a_x       (S_f - 1)p_f - a_x
    // Padding above            b_x       (S_f - 1)p_f + ((a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f)
    //                                    % q_x) - b_x
    // Data dilation strides    p_x       q_x
    // Output shape             S_o       S_x
    //
    // To _validate_, we simply need to check/infer the output shape of the forward convolution,
    // then check to make sure that the incoming delta has the same shape as the forward output.
    //
    // We will also compute and store the various parameters in the "backward" column above, since
    // some backends need them. (TODO(amprocte): Is it just because of the way the reference works
    // that this stuff is needed? If so, we can probably get rid of it and have conv_backprop
    // reference kernels that do the calculations of the backward parameters internally, or supply
    // utility functions to do it.)

    const PartialShape& filters_shape = get_input_partial_shape(0);
    element::Type filters_et = get_input_element_type(0);
    const PartialShape& delta_shape = get_input_partial_shape(1);
    element::Type delta_et = get_input_element_type(1);
    const PartialShape& bias_shape = get_input_partial_shape(2);
    element::Type bias_et = get_input_element_type(2);

    element::Type forward_result_et;
    PartialShape forward_result_shape;

    const PartialShape& fwd_filters_shape{
        filters_shape[1], filters_shape[0], filters_shape[2], filters_shape[3]};

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(forward_result_et, delta_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        delta_et,
        ", filters element type: ",
        filters_et,
        ").");

    forward_result_shape = infer_convolution_forward(this,
                                                     m_data_batch_shape,
                                                     m_data_dilation_strides_forward,
                                                     m_padding_below_forward,
                                                     m_padding_above_forward,
                                                     fwd_filters_shape,
                                                     m_window_movement_strides_forward,
                                                     m_window_dilation_strides_forward);
    NGRAPH_DEBUG << "\tpartial filter_shape: " << filters_shape << "delta_shape: " << delta_shape
                 << ", inferred_res_shape: " << forward_result_shape << endl;

    NODE_VALIDATION_CHECK(this,
                          forward_result_shape.compatible(delta_shape),
                          "Inferred forward output shape (",
                          forward_result_shape,
                          ") does not match shape of ",
                          "data_batch (",
                          m_data_batch_shape,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          filters_et.compatible(bias_et),
                          "Filter element type (",
                          filters_et,
                          ") does not match bias element type (",
                          bias_et,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          static_cast<size_t>(bias_shape.rank()) == 1,
                          "bias_shape size(",
                          bias_shape.rank(),
                          ") is not equal to 1");

    NODE_VALIDATION_CHECK(this,
                          static_cast<size_t>(bias_shape[0]) ==
                              static_cast<size_t>(filters_shape[0]),
                          "Filter input channel count (",
                          filters_shape,
                          ") does not compatible with ",
                          "bias shape channel count (",
                          bias_shape,
                          ").");

    set_output_type(0, forward_result_et, m_data_batch_shape);
}

void op::DeconvolutionBias::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                              const NodeVector& /* deltas */)
{
    throw ngraph_error("DeconvolutionBias generate_adjoints not supported implemented");
}

shared_ptr<Node> op::DeconvolutionBias::copy_with_new_args(const NodeVector& new_args) const
{
    NGRAPH_DEBUG << "DeconvolutionBias::copy_with_new_args" << endl;
    check_new_args_count(this, new_args);
    return make_shared<DeconvolutionBias>(m_data_batch_shape,
                                          new_args.at(0),
                                          new_args.at(1),
                                          new_args.at(2),
                                          m_window_movement_strides_forward,
                                          m_window_dilation_strides_forward,
                                          m_padding_below_forward,
                                          m_padding_above_forward,
                                          m_data_dilation_strides_forward,
                                          false);
}
