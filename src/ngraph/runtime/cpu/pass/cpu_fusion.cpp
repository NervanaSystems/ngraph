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

#include "cpu_fusion.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"

static bool init_cblas_arg(std::shared_ptr<ngraph::Node> reshape,
                           std::shared_ptr<ngraph::Node> arg,
                           bool& transpose_w,
                           ngraph::Shape& shape_w)
{
    auto r_w = std::dynamic_pointer_cast<ngraph::op::Reshape>(reshape);

    if (!r_w)
    {
        if (arg->get_shape().size() != 2)
        {
            NGRAPH_DEBUG << arg->get_name() << " 's rank != 2 "
                         << ngraph::vector_to_string(arg->get_shape());
            return false;
        }
        return true; //nth to do; reshape isn't a reshape
    }

    if (r_w->get_shape().size() != 2)
    {
        NGRAPH_DEBUG << "Reshape for " << reshape->get_name() << " doesn't reshape into matrix"
                     << ngraph::vector_to_string(r_w->get_shape());
        return false;
    }

    auto io = r_w->get_input_order();
    if (r_w->get_shape().size() != arg->get_shape().size()) //reshape
    {
        ngraph::AxisVector dio(io.size());
        std::iota(begin(dio), end(dio), 0);

        if (io != dio) //we can't reshape and transpose at the same time
        {
            NGRAPH_DEBUG << "Reshape for " << reshape->get_name() << " is not in default order "
                         << ngraph::vector_to_string(io);
            NGRAPH_DEBUG << "r_w shape = " << ngraph::vector_to_string(r_w->get_shape());
            NGRAPH_DEBUG << "arg shape = " << ngraph::vector_to_string(arg->get_shape());
            return false;
        }

        shape_w = r_w->get_shape();
    }
    else
    {
        if (io == ngraph::AxisVector{1, 0})
        {
            transpose_w = true;
        }
        //otherwise no-op reshape
    }

    return true;
}

template <typename T>
static std::vector<T> apply_permutation(std::vector<T> input, ngraph::AxisVector order)
{
    if (input.size() != order.size())
    {
        throw "input and order sizes don't match!";
    }

    std::vector<T> output(input.size());

    for (size_t i = 0; i < order.size(); i++)
    {
        output[i] = input.at(order.at(i));
    }

    return output;
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_matmulbias()
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto W = std::make_shared<pattern::op::Label>(element::f32, shape_w);
    auto x = std::make_shared<pattern::op::Label>(element::f32, shape_x);
    auto b = std::make_shared<pattern::op::Label>(element::f32, shape_b);

    auto pmmb = std::make_shared<op::MatmulBias>(
        W, x, nullptr, W->get_shape(), x->get_shape(), false, false);
    auto pbroadcast = std::make_shared<op::Broadcast>(b, pmmb->get_shape(), AxisSet{0});
    auto padd = pmmb + pbroadcast;

    ngraph::pattern::graph_rewrite_callback callback = [W, x](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_matmulbias_pattern against node = "
                     << m.get_match_root()->get_name();

        auto mpattern = m.get_match_root(); //add
        auto m_matmul = ngraph::pattern::Matcher::unique_match<op::MatmulBias>(mpattern);
        auto m_broadcast = ngraph::pattern::Matcher::unique_match<op::Broadcast>(mpattern);
        auto m_bias = m_broadcast->get_argument(0);
        auto pattern_map = m.get_pattern_map();

        auto mmb = std::make_shared<op::MatmulBias>(pattern_map[W],
                                                    pattern_map[x],
                                                    m_bias,
                                                    m_matmul->get_arg0_shape(),
                                                    m_matmul->get_arg1_shape(),
                                                    m_matmul->get_is_arg0_transposed(),
                                                    m_matmul->get_is_arg1_transposed(),
                                                    m_broadcast->get_broadcast_axes());

        ngraph::replace_node(m.get_match_root(), mmb);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(padd, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_matmul()
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    Shape shape_dot{2, 1};

    auto W = std::make_shared<pattern::op::Label>(element::f32, shape_w);
    auto x = std::make_shared<pattern::op::Label>(element::f32, shape_x);

    auto reshape_pred = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::Reshape>(n));
    };

    auto skip_w = std::make_shared<pattern::op::Skip>(W, reshape_pred);
    auto skip_x = std::make_shared<pattern::op::Skip>(x, reshape_pred);

    auto pdot = std::make_shared<op::Dot>(skip_w, skip_x);

    ngraph::pattern::graph_rewrite_callback callback = [W, x](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_matmul_pattern against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto mpattern = m.get_match_root();
        auto dot = m.get_match_root();

        if (mpattern->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << mpattern->get_name() << " type is not float!";
            return false;
        }

        if (dot->get_shape().size() != 2)
        {
            NGRAPH_DEBUG << "dot = " << dot->get_name() << " shape is not equal to 2!";
            return false;
        }

        if (shape_size(dot->get_shape()) == 0)
        {
            NGRAPH_DEBUG << "dot has a zero dimension";
            return false;
        }

        bool transpose_w = false;
        Shape shape_arg0{pattern_map[W]->get_shape()};
        if (!init_cblas_arg(dot->get_argument(0), pattern_map[W], transpose_w, shape_arg0))
        {
            return false;
        }

        bool transpose_x = false;
        Shape shape_arg1{pattern_map[x]->get_shape()};
        if (!init_cblas_arg(dot->get_argument(1), pattern_map[x], transpose_x, shape_arg1))
        {
            return false;
        }

        auto cg = std::shared_ptr<Node>(new op::MatmulBias(pattern_map[W],
                                                           pattern_map[x],
                                                           nullptr,
                                                           shape_arg0,
                                                           shape_arg1,
                                                           transpose_w,
                                                           transpose_x));

        ngraph::replace_node(mpattern, cg);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(pdot, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_fprop_bn()
{
    // construct varaiance
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::Multiply>(input, input);
    auto sum_input = std::make_shared<op::Sum>(input, AxisSet{0});
    auto square_sumed_input = std::make_shared<op::Multiply>(sum_input, sum_input);
    auto sum_squared_input = std::make_shared<op::Sum>(input_sq, AxisSet{0});
    auto avg_input_sum_sq = std::make_shared<op::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<op::Divide>(xmu, N);
    auto variance_label =
        std::make_shared<pattern::op::Label>(variance, nullptr, NodeVector{variance});
    auto variance_with_broadcast =
        std::make_shared<op::Broadcast>(variance_label, Shape{2, 3}, AxisSet{0});

    // construct mean
    auto sum_input1 = std::make_shared<op::Sum>(input, AxisSet{0});
    auto mean = std::make_shared<op::Divide>(sum_input1, N);
    auto mean_label = std::make_shared<pattern::op::Label>(mean, nullptr, NodeVector{mean});
    auto mean_with_broadcast = std::make_shared<op::Broadcast>(mean_label, Shape{2, 3}, AxisSet{0});
    auto input_diff_mean = std::make_shared<op::Subtract>(input, mean_with_broadcast);

    // Eps
    auto eps_label = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
    auto eps_with_broadcast = std::make_shared<op::Broadcast>(eps_label, Shape{2, 3}, AxisSet{0});

    auto add1 = std::make_shared<op::Add>(eps_with_broadcast, variance_with_broadcast);
    auto sqrt_variance_eps = std::make_shared<op::Sqrt>(add1);
    auto divide_mean_variance = std::make_shared<op::Divide>(input_diff_mean, sqrt_variance_eps);

    //Gamma
    auto gamma_label = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
    auto gamma_with_broadcast =
        std::make_shared<op::Broadcast>(gamma_label, Shape{2, 3}, AxisSet{0});
    auto multiply_gamma =
        std::make_shared<op::Multiply>(gamma_with_broadcast, divide_mean_variance);

    //Beta
    auto beta_label = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
    auto beta_with_broadcast = std::make_shared<op::Broadcast>(beta_label, Shape{2, 3}, AxisSet{0});

    auto add_beta = std::make_shared<op::Add>(beta_with_broadcast, multiply_gamma);
    // This completes fprop bn pattern

    //Define a call back that needs to called once the DFG matches the pattern
    ngraph::pattern::graph_rewrite_callback callback =
        [variance_label, mean_label, input, eps_label, gamma_label, beta_label](
            pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_fprop_bn pattern against "
                         << m.get_match_root()->get_name();

            //TODO - add assert's based on the matched node
            auto pattern_map = m.get_pattern_map();
            NGRAPH_DEBUG << "Input: " << pattern_map[input]->get_name() << " "
                         << pattern_map[input]->get_shape().size();
            NGRAPH_DEBUG << "Variance: " << pattern_map[variance_label]->get_name() << " "
                         << pattern_map[variance_label]->get_shape().size();
            NGRAPH_DEBUG << "Mean: " << pattern_map[mean_label]->get_name() << " "
                         << pattern_map[mean_label]->get_shape().size();
            NGRAPH_DEBUG << "eps: " << pattern_map[eps_label]->get_name() << " "
                         << pattern_map[eps_label]->get_shape().size();
            NGRAPH_DEBUG << "gamma: " << pattern_map[gamma_label]->get_name() << " "
                         << pattern_map[gamma_label]->get_shape().size();
            NGRAPH_DEBUG << "beta: " << pattern_map[beta_label]->get_name() << " "
                         << pattern_map[beta_label]->get_shape().size();

            // dont fuse if the inout doesnt have 4dims
            if (pattern_map[input]->get_shape().size() != 4)
            {
                NGRAPH_DEBUG << "Input to bn doesnt not have a rank=4, so not fusing";
                return false;
            }
            Shape bn_output_shape{m.get_match_root()->get_shape()};
            Shape m_bn_mean_shape{pattern_map[mean_label]->get_shape()};
            Shape m_bn_variance_shape{pattern_map[variance_label]->get_shape()};

            // get epsilon value
            auto eps_ptr = std::dynamic_pointer_cast<op::Constant>(pattern_map[eps_label]);
            double epsilon = *(reinterpret_cast<const double*>(eps_ptr->get_data_ptr()));
            auto bn_node = std::make_shared<op::BatchNorm>(
                epsilon, pattern_map[gamma_label], pattern_map[beta_label], pattern_map[input]);

            auto normalized_output = std::shared_ptr<Node>(new op::GetOutputElement(bn_node, 0));

            ngraph::replace_node(m.get_match_root(), normalized_output);
            return true;
        };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add_beta, callback);
    this->add_matcher(m);
}

static bool
    zero_padded_conv_consistency_check(const std::shared_ptr<ngraph::Node>& match_root,
                                       const std::shared_ptr<ngraph::op::Constant>& pad_value_op,
                                       const std::shared_ptr<ngraph::Node>& pad_input,
                                       const std::shared_ptr<ngraph::op::Pad>& matched_pad,
                                       const ngraph::CoordinateDiff& padding_below,
                                       const ngraph::CoordinateDiff& padding_above,
                                       size_t batch_index,
                                       size_t channel_index)
{
    // Only match float32 convolutions
    if (match_root->get_element_type() != ngraph::element::f32)
    {
        return false;
    }

    // Only match zero padding
    if (pad_value_op->get_vector<float>().at(0) != 0.0f)
    {
        return false;
    }

    // Only match 4D tensors
    if (pad_input->get_shape().size() != 4)
    {
        return false;
    }

    // Only match no interior padding
    if (matched_pad->get_padding_interior() != ngraph::Shape(pad_input->get_shape().size()))
    {
        return false;
    }

    // Only match convolutions with no padding specification
    if (padding_below != ngraph::CoordinateDiff(2) || padding_above != ngraph::CoordinateDiff(2))
    {
        return false;
    }

    // Only match no padding in the batch dimension
    if (matched_pad->get_padding_above().at(batch_index) != 0 ||
        matched_pad->get_padding_below().at(batch_index) != 0)
    {
        return false;
    }

    // Only match no padding in the channel dimension
    if (matched_pad->get_padding_above().at(channel_index) != 0 ||
        matched_pad->get_padding_below().at(channel_index) != 0)
    {
        return false;
    }

    return true;
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_zero_padded_reshaped_conv()
{
    auto pad_input = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto pad_value = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto pad = std::make_shared<op::Pad>(pad_input, pad_value, Shape{}, Shape{}, Shape{});
    auto pad_label = std::make_shared<pattern::op::Label>(pad, nullptr, NodeVector{pad});

    auto reshape = std::make_shared<op::Reshape>(pad_label, AxisVector{}, Shape{1, 1, 1, 1});
    auto reshape_label =
        std::make_shared<pattern::op::Label>(reshape, nullptr, NodeVector{reshape});

    auto conv_filter = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});

    auto conv = std::make_shared<op::Convolution>(reshape_label,
                                                  conv_filter,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{1, 1},
                                                  CoordinateDiff{1, 1},
                                                  Strides{1, 1});
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, NodeVector{conv});

    ngraph::pattern::graph_rewrite_callback callback =
        [pad_input, pad_value, pad_label, reshape_label, conv_filter, conv_label](
            pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_map();

            auto pad_value_op = std::dynamic_pointer_cast<op::Constant>(pattern_map[pad_value]);

            const auto& matched_conv =
                std::dynamic_pointer_cast<op::Convolution>(pattern_map[conv_label]);
            const auto& matched_pad = std::dynamic_pointer_cast<op::Pad>(pattern_map[pad_label]);
            const auto& matched_reshape =
                std::dynamic_pointer_cast<op::Reshape>(pattern_map[reshape_label]);

            const auto& input_order = matched_reshape->get_input_order();
            auto hoisted_reshape_output_shape = apply_permutation<Shape::value_type>(
                pattern_map[pad_input]->get_shape(), input_order);

            auto hoisted_reshape = std::make_shared<op::Reshape>(
                pattern_map[pad_input],
                input_order,
                Shape(hoisted_reshape_output_shape.begin(), hoisted_reshape_output_shape.end()));

            if (!zero_padded_conv_consistency_check(m.get_match_root(),
                                                    pad_value_op,
                                                    pattern_map[pad_input],
                                                    matched_pad,
                                                    matched_conv->get_padding_below(),
                                                    matched_conv->get_padding_above(),
                                                    input_order[0],
                                                    input_order[1]))
            {
                return false;
            }

            CoordinateDiff padding_below{static_cast<CoordinateDiff::value_type>(
                                             matched_pad->get_padding_below().at(input_order[2])),
                                         static_cast<CoordinateDiff::value_type>(
                                             matched_pad->get_padding_below().at(input_order[3]))};
            CoordinateDiff padding_above{static_cast<CoordinateDiff::value_type>(
                                             matched_pad->get_padding_above().at(input_order[2])),
                                         static_cast<CoordinateDiff::value_type>(
                                             matched_pad->get_padding_above().at(input_order[3]))};

            auto zero_padded_conv =
                std::make_shared<op::Convolution>(hoisted_reshape,
                                                  pattern_map[conv_filter],
                                                  matched_conv->get_window_movement_strides(),
                                                  matched_conv->get_window_dilation_strides(),
                                                  padding_below,
                                                  padding_above,
                                                  matched_conv->get_data_dilation_strides());

            ngraph::replace_node(m.get_match_root(), zero_padded_conv);
            return true;
        };

    this->add_matcher(std::make_shared<ngraph::pattern::Matcher>(conv_label, callback));
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_zero_padded_conv()
{
    auto pad_input = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto pad_value = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto pad = std::make_shared<op::Pad>(
        pad_input, pad_value, Shape{0, 0, 0, 0}, Shape{0, 0, 0, 0}, Shape{0, 0, 0, 0});
    auto pad_label = std::make_shared<pattern::op::Label>(pad, nullptr, NodeVector{pad});

    auto conv_filter = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});

    auto conv = std::make_shared<op::Convolution>(pad_label,
                                                  conv_filter,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{1, 1},
                                                  CoordinateDiff{1, 1},
                                                  Strides{1, 1});
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, NodeVector{conv});

    ngraph::pattern::graph_rewrite_callback callback =
        [pad_input, pad_value, pad_label, conv_filter, conv_label](pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_map();

            auto pad_value_op = std::dynamic_pointer_cast<op::Constant>(pattern_map[pad_value]);

            const auto& matched_conv =
                std::dynamic_pointer_cast<op::Convolution>(pattern_map[conv_label]);
            const auto& matched_pad = std::dynamic_pointer_cast<op::Pad>(pattern_map[pad_label]);

            if (!zero_padded_conv_consistency_check(m.get_match_root(),
                                                    pad_value_op,
                                                    pattern_map[pad_input],
                                                    matched_pad,
                                                    matched_conv->get_padding_below(),
                                                    matched_conv->get_padding_above(),
                                                    0,
                                                    1))
            {
                return false;
            }

            CoordinateDiff padding_below{
                static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_below().at(2)),
                static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_below().at(3))};
            CoordinateDiff padding_above{
                static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_above().at(2)),
                static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_above().at(3))};

            auto zero_padded_conv =
                std::make_shared<op::Convolution>(pattern_map[pad_input],
                                                  pattern_map[conv_filter],
                                                  matched_conv->get_window_movement_strides(),
                                                  matched_conv->get_window_dilation_strides(),
                                                  padding_below,
                                                  padding_above,
                                                  matched_conv->get_data_dilation_strides());

            ngraph::replace_node(m.get_match_root(), zero_padded_conv);
            return true;
        };

    this->add_matcher(std::make_shared<ngraph::pattern::Matcher>(conv_label, callback));
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_zero_padded_conv_backprop_filters()
{
    auto pad_input = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto pad_value = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto pad = std::make_shared<op::Pad>(
        pad_input, pad_value, Shape{0, 0, 0, 0}, Shape{0, 0, 0, 0}, Shape{0, 0, 0, 0});
    auto pad_label = std::make_shared<pattern::op::Label>(pad, nullptr, NodeVector{pad});

    auto output_delta = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});

    auto conv = std::make_shared<op::ConvolutionBackpropFilters>(pad_label,
                                                                 Shape{1, 1, 3, 3},
                                                                 output_delta,
                                                                 Strides{1, 1},
                                                                 Strides{1, 1},
                                                                 CoordinateDiff{1, 1},
                                                                 CoordinateDiff{1, 1},
                                                                 Strides{1, 1});
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, NodeVector{conv});

    ngraph::pattern::graph_rewrite_callback callback =
        [pad_input, pad_value, pad_label, output_delta, conv_label](pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_map();

            auto pad_value_op = std::dynamic_pointer_cast<op::Constant>(pattern_map[pad_value]);

            const auto& matched_conv =
                std::dynamic_pointer_cast<op::ConvolutionBackpropFilters>(pattern_map[conv_label]);
            const auto& matched_pad = std::dynamic_pointer_cast<op::Pad>(pattern_map[pad_label]);

            if (!zero_padded_conv_consistency_check(m.get_match_root(),
                                                    pad_value_op,
                                                    pattern_map[pad_input],
                                                    matched_pad,
                                                    matched_conv->get_padding_below_forward(),
                                                    matched_conv->get_padding_above_forward(),
                                                    0,
                                                    1))
            {
                return false;
            }

            CoordinateDiff padding_below{
                static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_below().at(2)),
                static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_below().at(3))};
            CoordinateDiff padding_above{
                static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_above().at(2)),
                static_cast<CoordinateDiff::value_type>(matched_pad->get_padding_above().at(3))};

            auto zero_padded_conv_backprop_filters =
                std::make_shared<op::ConvolutionBackpropFilters>(
                    pattern_map[pad_input],
                    matched_conv->get_filters_shape(),
                    pattern_map[output_delta],
                    matched_conv->get_window_movement_strides_forward(),
                    matched_conv->get_window_dilation_strides_forward(),
                    padding_below,
                    padding_above,
                    matched_conv->get_data_dilation_strides_forward());

            ngraph::replace_node(m.get_match_root(), zero_padded_conv_backprop_filters);
            return true;
        };

    this->add_matcher(std::make_shared<ngraph::pattern::Matcher>(conv_label, callback));
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_sigmoid()
{
    //construct variance
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_input = std::make_shared<op::Negative>(input);
    auto exp_neg_input = std::make_shared<op::Exp>(neg_input);

    // broadcast input
    auto constant = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto broadcast_constant = std::make_shared<op::Broadcast>(constant, Shape{3, 4}, AxisSet{0, 1});

    auto add_exp = std::make_shared<op::Add>(exp_neg_input, broadcast_constant);
    auto divide_1_over_exp = std::make_shared<op::Divide>(broadcast_constant, add_exp);

    //Define a call back that needs to called once the DFG matches the pattern
    ngraph::pattern::graph_rewrite_callback callback = [input](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_fprop_sigmoid pattern against "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        if (m.get_match_root()->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << " type is not float!";
            return false;
        }

        if (m.get_match_root()->get_outputs().size() != pattern_map[input]->get_outputs().size())
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << "input= " << pattern_map[input]->get_name() << "size dont match!";
            return false;
        }

        auto sigmoid_node = std::make_shared<op::Sigmoid>(pattern_map[input]);
        ngraph::replace_node(m.get_match_root(), sigmoid_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide_1_over_exp, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_sigmoid_bprop()
{
    //construct variance
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_input = std::make_shared<op::Negative>(input);
    auto exp_neg_input = std::make_shared<op::Exp>(neg_input);

    // broadcast input
    auto constant = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto broadcast_constant = std::make_shared<op::Broadcast>(constant, Shape{3, 4}, AxisSet{0, 1});

    auto add_exp = std::make_shared<op::Add>(exp_neg_input, broadcast_constant);
    // //auto divide_1_over_exp = std::make_shared<op::Divide>(broadcast_constant, add_exp);
    auto sigmoid_fwd = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});

    auto delta = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_delta = std::make_shared<op::Negative>(delta);

    auto multiply_sigmoid_delta = std::make_shared<op::Multiply>(sigmoid_fwd, neg_delta);
    auto divide_2 = std::make_shared<op::Divide>(multiply_sigmoid_delta, add_exp);

    auto multiply_2 = std::make_shared<op::Multiply>(divide_2, exp_neg_input);
    auto negtive_2 = std::make_shared<op::Negative>(multiply_2);

    //Define a call back that needs to called once the DFG matches the pattern
    ngraph::pattern::graph_rewrite_callback callback = [input, delta](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_fprop_sigmoid pattern against "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        if (m.get_match_root()->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << " type is not float!";
            return false;
        }

        if (m.get_match_root()->get_shape().size() != pattern_map[input]->get_shape().size())
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << "input= " << pattern_map[input]->get_name() << "size dont match!";
            return false;
        }
        auto dsigmoid =
            std::make_shared<op::SigmoidBackprop>(pattern_map[input], pattern_map[delta]);
        ngraph::replace_node(m.get_match_root(), dsigmoid);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(negtive_2, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_bias()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto pbias = std::make_shared<pattern::op::Label>(element::f32, Shape{});

    auto pbroadcast = std::make_shared<op::Broadcast>(pbias, shape, AxisSet{0, 1, 2, 3});

    auto pconv1 = std::make_shared<op::Convolution>(data_batch,
                                                    filters,
                                                    Strides{1, 1},
                                                    Strides{1, 1},
                                                    CoordinateDiff{0, 0},
                                                    CoordinateDiff{0, 0},
                                                    Strides{1, 1});
    auto p_conv_bias = pbroadcast + pconv1;

    ngraph::pattern::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_conv_bias against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto conv = std::dynamic_pointer_cast<op::Convolution>(m.get_match_root()->get_argument(0));
        if (conv->get_input_shape(0).size() == 4)
        {
            auto bias = m.get_match_root()->get_argument(1)->get_argument(0);
            auto bias_shape = bias->get_shape();
            if (bias_shape.size() > 1)
            {
                NGRAPH_DEBUG
                    << "mpattern = " << m.get_match_root()->get_name()
                    << "conv_bias bias shape != 1, requires reshape to match filter count.";
                ngraph::AxisVector order(bias_shape.size());
                std::iota(begin(order), end(order), 0);
                auto bias_reshape =
                    std::make_shared<op::Reshape>(bias, order, Shape{conv->get_input_shape(1)[0]});
                auto conv_bias = std::shared_ptr<Node>(new op::ConvolutionBias(conv, bias_reshape));
                ngraph::replace_node(m.get_match_root(), conv_bias);
                return true;
            }
            else
            {
                auto conv_bias = std::shared_ptr<Node>(new op::ConvolutionBias(conv, bias));
                ngraph::replace_node(m.get_match_root(), conv_bias);
                return true;
            }
        }
        NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                     << "conv_bias fusion skipped due to input rank size != 4.";
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_conv_bias, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_batch_norm_relu()
{
    auto input_shape = Shape{1, 2, 2, 2};
    auto input = std::make_shared<pattern::op::Label>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto var_shape = Shape{2};
    auto gamma_shape = Shape{2};
    auto gamma = std::make_shared<pattern::op::Label>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = std::make_shared<pattern::op::Label>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = std::make_shared<op::BatchNorm>(eps, gamma, beta, input);
    auto goe = std::make_shared<op::GetOutputElement>(bn, 0);
    auto prelu = std::make_shared<op::Relu>(goe);

    ngraph::pattern::graph_rewrite_callback callback = [input, gamma, beta](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_batch_norm_relu against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto m_bn = std::dynamic_pointer_cast<op::BatchNorm>(
            m.get_match_root()->get_argument(0)->get_inputs().at(0).get_output().get_node());

        //as of now, only MKLDNN supports this fusion
        //and it requires input data's rank to be equal to 4
        if (pattern_map[input]->get_shape().size() != 4)
        {
            NGRAPH_DEBUG << " Input data's rank isn't equal to 4. Shape = "
                         << pattern_map[input]->get_shape().size();
            return false;
        }

        std::vector<std::shared_ptr<Node>> mgoes(m_bn->get_outputs().size());
        for (auto bn_in : m_bn->get_output_inputs(0))
        {
            auto mgoe = std::dynamic_pointer_cast<op::GetOutputElement>(bn_in->get_node());
            mgoes[mgoe->get_n()] = mgoe;
        }

        if (mgoes[0]->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Relu isn't the only user of BatchNorm's output";
            return false;
        }

        mgoes[0] = m.get_match_root(); //replace relu instead of its GetOutputElement

        auto bn_relu = std::make_shared<op::BatchNormRelu>(
            m_bn->get_eps_value(), pattern_map[gamma], pattern_map[beta], pattern_map[input]);

        auto bn_relu_output = std::make_shared<op::GetOutputElement>(bn_relu, 0);
        auto bn_relu_mean = std::make_shared<op::GetOutputElement>(bn_relu, 1);
        auto bn_relu_var = std::make_shared<op::GetOutputElement>(bn_relu, 2);

        std::shared_ptr<Node> new_nodes[] = {bn_relu_output, bn_relu_mean, bn_relu_var};

        for (size_t i = 0; i < mgoes.size(); i++)
        {
            ngraph::replace_node(mgoes.at(i), new_nodes[i]);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_batch_norm_relu_global_stats()
{
    auto input_shape = Shape{1, 2, 2, 2};
    auto input = std::make_shared<pattern::op::Label>(element::f32, input_shape);
    auto mean_shape = Shape{2};
    auto mean = std::make_shared<pattern::op::Label>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = std::make_shared<pattern::op::Label>(element::f32, var_shape);
    auto gamma_shape = Shape{2};
    auto gamma = std::make_shared<pattern::op::Label>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = std::make_shared<pattern::op::Label>(element::f32, beta_shape);
    double eps = 0.001;
    auto shape_r = Shape{1, 2, 2, 2};
    auto bn = std::make_shared<op::BatchNorm>(eps, gamma, beta, input, mean, var);
    auto prelu = std::make_shared<op::Relu>(bn);

    ngraph::pattern::graph_rewrite_callback callback =
        [input, mean, var, gamma, beta](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In callback for construct_batch_norm_relu against node = "
                         << m.get_match_root()->get_name();

            auto pattern_map = m.get_pattern_map();
            auto m_bn = std::dynamic_pointer_cast<op::BatchNorm>(
                m.get_match_root()->get_inputs().at(0).get_output().get_node());

            //as of now, only MKLDNN supports this fusion
            //and it requires input data's rank to be equal to 4
            if (pattern_map[input]->get_shape().size() != 4)
            {
                NGRAPH_DEBUG << " Input data's rank isn't equal to 4. Shape = "
                             << pattern_map[input]->get_shape().size();
                return false;
            }

            if (m_bn->get_users().size() > 1)
            {
                NGRAPH_DEBUG << "Relu isn't the only user of BatchNorm's output";
                return false;
            }

            auto bn_relu = std::make_shared<op::BatchNormRelu>(m_bn->get_eps_value(),
                                                               pattern_map[gamma],
                                                               pattern_map[beta],
                                                               pattern_map[input],
                                                               pattern_map[mean],
                                                               pattern_map[var],
                                                               m_bn->get_training_flag());

            ngraph::replace_node(m.get_match_root(), bn_relu);

            return true;
        };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_relu()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);

    auto pconv = std::make_shared<op::Convolution>(data_batch,
                                                   filters,
                                                   Strides{1, 1},
                                                   Strides{1, 1},
                                                   CoordinateDiff{0, 0},
                                                   CoordinateDiff{0, 0},
                                                   Strides{1, 1});

    auto prelu = std::make_shared<op::Relu>(pconv);

    pattern::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_relu against "
                     << m.get_match_root()->get_name();

        auto conv = std::dynamic_pointer_cast<op::Convolution>(m.get_match_root()->get_argument(0));

        //These checks are to make sure a MKLDNN Convolution kernel can be used.
        bool data_dilated = false;
        for (size_t s : conv->get_data_dilation_strides())
        {
            data_dilated = data_dilated || (s != 1);
        }

        if (data_dilated)
        {
            NGRAPH_DEBUG << "Convolution has dilations greater than 1";
            return false;
        }

        if (conv->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "Convolution isn't of type float";
            return false;
        }

        auto arg0_rank = conv->get_input_shape(0).size();
        auto arg1_rank = conv->get_input_shape(1).size();

        if (arg0_rank != 4 || arg1_rank != 4)
        {
            NGRAPH_DEBUG << "Convolution's arguments ranks aren't equal to 4";
            return false;
        }

        if (conv->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Convolution has more than one user";
            return false;
        }

        auto conv_relu = std::shared_ptr<Node>(new op::ConvolutionRelu(conv));
        ngraph::replace_node(m.get_match_root(), conv_relu);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(prelu, callback);
    this->add_matcher(m);
}
