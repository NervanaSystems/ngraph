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

#include "deconv.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

void op::util::validate_deconvbias_shapes(const Shape& data_shape,
                                        const Shape& filters_shape,
                                        const Shape& bias_shape)
{
    if (bias_shape.size() != 1)
    {
        throw ngraph_error("Deconvolution+bias bias is expected to be 1D, but has shape: " +
                           vector_to_string(bias_shape));
    }
    if (bias_shape[0] != filters_shape[0])
    {
        throw ngraph_error(
            "Deconvolution+bias bias element size does not match number of filters. bias_size = " +
            std::to_string(bias_shape[0]) + ", num_filters = " + std::to_string(filters_shape[0]));
    }
    if (data_shape[1] != filters_shape[1])
    {
        throw ngraph_error(
            "Deconvolution+bias data and filter have different number of channels: data_channel=" +
            std::to_string(data_shape[1]) + ", filter_channel= " +
            std::to_string(filters_shape[1]));
    }
}

op::DeconvolutionBias::DeconvolutionBias(const Shape& data_batch_shape,
                                        const shared_ptr<Node>& filters,
                                        const shared_ptr<Node>& output_delta,
                                        const shared_ptr<Node>& bias,
                                        const Strides& window_movement_strides_forward,
                                        const Strides& window_dilation_strides_forward,
                                        const CoordinateDiff& padding_below_forward,
                                        const CoordinateDiff& padding_above_forward,
                                        const Strides& data_dilation_strides_forward,
                                        const bool with_relu)
    : Op("DeconvolutionBias", check_single_output_args({filters, output_delta, bias}))
    , m_data_batch_shape(data_batch_shape)
    , m_window_movement_strides_forward(window_movement_strides_forward)
    , m_window_dilation_strides_forward(window_dilation_strides_forward)
    , m_padding_below_forward(padding_below_forward)
    , m_padding_above_forward(padding_above_forward)
    , m_data_dilation_strides_forward(data_dilation_strides_forward)
    , m_with_relu(with_relu)
{
    NGRAPH_DEBUG << "DeconvolutionBias ctor" << endl;
    NGRAPH_DEBUG << "data: " << data_batch_shape << ", filters: " << filters->get_shape()
                 << ", output_delta: " << output_delta->get_shape();
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
    // Padding above            b_x       (S_f - 1)p_f + ((a_x + (S_x - 1)p_x + b_x - (S_f - 1)p_f) % q_x) - b_x
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

    std::tie(forward_result_et, forward_result_shape) =
        infer_convolution_forward(this,
                                  delta_et,
                                  filters_et,
                                  m_data_batch_shape,
                                  m_data_dilation_strides_forward,
                                  m_padding_below_forward,
                                  m_padding_above_forward,
                                  filters_shape,
                                  m_window_movement_strides_forward,
                                  m_window_dilation_strides_forward);
    NGRAPH_DEBUG << "\tpartial filter_shape: " << filters_shape << "delta_shape: " << delta_shape
                 << ", inferred_res_shape: " << forward_result_shape << endl ;

    NODE_VALIDATION_ASSERT(this, forward_result_shape.compatible(delta_shape))
        << "Inferred forward output shape (" << forward_result_shape << ") does not match shape of "
        << "delta (" << delta_shape << ").";

    set_output_type(0, forward_result_et, m_data_batch_shape);

    NODE_VALIDATION_ASSERT(this, delta_shape.compatible(bias_shape))
        << "Filter input channel count (" << delta_shape << ") does not compatible with "
        << "bias shape channel count (" << bias_shape << ").";

    NODE_VALIDATION_ASSERT(this, filters_et.compatible(bias_et))
        << "Filter element type (" << filters_et << ") does not match bias element type ("
        << bias_et << ").";
    //
    // Compute parameters needed for backprop-as-convolution.
    //
    // TODO(amprocte): Remove these fields, compute where needed.
    //
    if (delta_shape.is_static() && filters_shape.is_static())
    {
        size_t spatial_dim_count = static_cast<size_t>(delta_shape.rank()) - 2;

        m_window_movement_strides_backward = m_data_dilation_strides_forward;
        m_window_dilation_strides_backward = m_window_dilation_strides_forward;
        m_data_dilation_strides_backward = m_window_movement_strides_forward;

        m_padding_below_backward.resize(spatial_dim_count);
        m_padding_above_backward.resize(spatial_dim_count);

        for (size_t i = 0; i < spatial_dim_count; i++)
        {
            m_padding_below_backward[i] = (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                                              m_window_dilation_strides_forward[i] -
                                          m_padding_below_forward[i];
            m_padding_above_backward[i] =
                (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                    m_window_dilation_strides_forward[i] +
                ((m_padding_below_forward[i] +
                  (m_data_batch_shape[i + 2] - 1) * m_data_dilation_strides_forward[i] +
                  m_padding_above_forward[i] -
                  (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) *
                      m_window_dilation_strides_forward[i]) %
                 m_window_movement_strides_forward[i]) -
                m_padding_above_forward[i];
        }
    }
}

void op::DeconvolutionBias::generate_adjoints(autodiff::Adjoints& adjoints,
                                                    const NodeVector& deltas)
{
    throw ngraph_error("DeconvolutionBias generate_adjoints not supported implemented");

    /*NGRAPH_DEBUG << "DeconvolutionBias::generate_adjoints" << endl;
    auto delta = deltas.at(0);

    auto x = get_argument(1);
    const auto x_shape = x->get_shape();

    auto f = get_argument(0);
    const auto f_shape = f->get_shape();

    auto data_conv = make_shared<op::Convolution>(delta,
                                                  f,
                                                  m_window_movement_strides_forward,
                                                  m_window_dilation_strides_forward,
                                                  m_padding_below_forward,
                                                  m_padding_above_forward,
                                                  m_data_dilation_strides_forward);

    adjoints.add_delta(x, data_conv);

    Strides window_movement_strides;
    Strides window_dilation_strides;
    CoordinateDiff padding_below;
    CoordinateDiff padding_above;
    Strides data_dilation_strides;
    for (size_t i = 0; i < f_shape.size() - 2; i++)
    {
        window_movement_strides.push_back(m_window_dilation_strides_backward[i]);
        window_dilation_strides.push_back(m_window_movement_strides_backward[i]);
        padding_below.push_back(m_padding_below_backward[i]);
        padding_above.push_back(m_padding_above_backward[i] -
                                (m_padding_below_backward[i] +
                                 (x_shape[i + 2] - 1) * m_data_dilation_strides_backward[i] +
                                 m_padding_above_backward[i] -
                                 (f_shape[i + 2] - 1) * m_window_dilation_strides_backward[i]) %
                                    m_window_movement_strides_backward[i]);
        data_dilation_strides.push_back(m_data_dilation_strides_backward[i]);
    }

    auto swap_NC = [](const shared_ptr<Node> n) {
        AxisVector ax_order = ngraph::get_default_order(n->get_shape());
        ax_order[0] = 1;
        ax_order[1] = 0;

        auto new_shape = n->get_shape();
        new_shape[0] = n->get_shape()[1];
        new_shape[1] = n->get_shape()[0];

        return make_shared<op::Reshape>(n, ax_order, new_shape);
    };

    delta = swap_NC(delta);
    x = swap_NC(x);

    shared_ptr<Node> filter_deconv_bprop = make_shared<op::Convolution>(x,
                                                                        delta,
                                                                        window_movement_strides,
                                                                        window_dilation_strides,
                                                                        padding_below,
                                                                        padding_above,
                                                                        data_dilation_strides);
    AxisSet axes;
    for (size_t i = 2; i < filter_deconv_bprop->get_shape().size(); ++i)
    {
        axes.insert(i);
    }
    filter_deconv_bprop = make_shared<ngraph::op::Reverse>(filter_deconv_bprop, axes);
    adjoints.add_delta(f, filter_deconv_bprop);*/
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
                                                false); //TODO: check
}
