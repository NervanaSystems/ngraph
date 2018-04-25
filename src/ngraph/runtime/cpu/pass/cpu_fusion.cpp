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
#include <typeindex>
#include <typeinfo>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include "cpu_fusion.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
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
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
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
                     << m.match_root()->get_name();

        auto mpattern = m.match_root(); //add
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

        ngraph::replace_node(m.match_root(), mmb);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(padd, callback);
    std::cout << "MatmulBias: " << m << std::endl;
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

    auto skip_w = std::make_shared<pattern::op::Any>(W, reshape_pred);
    auto skip_x = std::make_shared<pattern::op::Any>(x, reshape_pred);

    auto pdot = std::make_shared<op::Dot>(skip_w, skip_x);

    ngraph::pattern::graph_rewrite_callback callback = [W, x](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_matmul_pattern against node = "
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto mpattern = m.match_root();
        auto dot = m.match_root();

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
    std::cout << "Matmul: " << m << std::endl;
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
                         << m.match_root()->get_name();

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
            Shape bn_output_shape{m.match_root()->get_shape()};
            Shape m_bn_mean_shape{pattern_map[mean_label]->get_shape()};
            Shape m_bn_variance_shape{pattern_map[variance_label]->get_shape()};

            // get epsilon value
            auto eps_ptr = std::dynamic_pointer_cast<op::Constant>(pattern_map[eps_label]);
            double epsilon = *(reinterpret_cast<const double*>(eps_ptr->get_data_ptr()));
            auto bn_node = std::make_shared<op::BatchNorm>(
                epsilon, pattern_map[gamma_label], pattern_map[beta_label], pattern_map[input]);

            auto normalized_output = std::shared_ptr<Node>(new op::GetOutputElement(bn_node, 0));

            ngraph::replace_node(m.match_root(), normalized_output);
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

            if (!zero_padded_conv_consistency_check(m.match_root(),
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

            ngraph::replace_node(m.match_root(), zero_padded_conv);
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

            if (!zero_padded_conv_consistency_check(m.match_root(),
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

            ngraph::replace_node(m.match_root(), zero_padded_conv);
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

            if (!zero_padded_conv_consistency_check(m.match_root(),
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

            ngraph::replace_node(m.match_root(), zero_padded_conv_backprop_filters);
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
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        if (m.match_root()->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name() << " type is not float!";
            return false;
        }

        if (m.match_root()->get_outputs().size() != pattern_map[input]->get_outputs().size())
        {
            NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name()
                         << "input= " << pattern_map[input]->get_name() << "size dont match!";
            return false;
        }

        auto sigmoid_node = std::make_shared<op::Sigmoid>(pattern_map[input]);
        ngraph::replace_node(m.match_root(), sigmoid_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide_1_over_exp, callback);
    std::cout << "Sigmoid: " << m << std::endl;
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
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        if (m.match_root()->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name() << " type is not float!";
            return false;
        }

        if (m.match_root()->get_shape().size() != pattern_map[input]->get_shape().size())
        {
            NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name()
                         << "input= " << pattern_map[input]->get_name() << "size dont match!";
            return false;
        }
        auto dsigmoid =
            std::make_shared<op::SigmoidBackprop>(pattern_map[input], pattern_map[delta]);
        ngraph::replace_node(m.match_root(), dsigmoid);
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
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto conv = std::dynamic_pointer_cast<op::Convolution>(m.match_root()->get_argument(0));
        if (conv->get_input_shape(0).size() == 4)
        {
            auto bias = m.match_root()->get_argument(1)->get_argument(0);
            auto bias_shape = bias->get_shape();
            if (bias_shape.size() > 1)
            {
                NGRAPH_DEBUG
                    << "mpattern = " << m.match_root()->get_name()
                    << "conv_bias bias shape != 1, requires reshape to match filter count.";
                ngraph::AxisVector order(bias_shape.size());
                std::iota(begin(order), end(order), 0);
                auto bias_reshape =
                    std::make_shared<op::Reshape>(bias, order, Shape{conv->get_input_shape(1)[0]});
                auto conv_bias = std::shared_ptr<Node>(new op::ConvolutionBias(conv, bias_reshape));
                ngraph::replace_node(m.match_root(), conv_bias);
                return true;
            }
            else
            {
                auto conv_bias = std::shared_ptr<Node>(new op::ConvolutionBias(conv, bias));
                ngraph::replace_node(m.match_root(), conv_bias);
                return true;
            }
        }
        NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name()
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
                     << m.match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto m_bn = std::dynamic_pointer_cast<op::BatchNorm>(
            m.match_root()->get_argument(0)->get_inputs().at(0).get_output().get_node());

        if (!m_bn->get_training_flag())
        {
            NGRAPH_DEBUG << " This is an inference batchnorm, so skipping fusion";
            return false;
        }

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

        mgoes[0] = m.match_root(); //replace relu instead of its GetOutputElement

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
                     << m.match_root()->get_name();

        auto conv = std::dynamic_pointer_cast<op::Convolution>(m.match_root()->get_argument(0));

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
        ngraph::replace_node(m.match_root(), conv_relu);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(prelu, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_lstm_fprop()
{
    // param1_1 -> ht_1 (src_iter)
    auto param1_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto broadcast_pred_1 = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::Broadcast>(n));
    };
    auto skip_param_1_1 = std::make_shared<pattern::op::Any>(param1_1, broadcast_pred_1);
    // param1_2 -> h2h weights (weights_iter)
    auto param1_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto param1_2_reshape =
        std::make_shared<op::Reshape>(param1_2, AxisVector{1, 0}, Shape{100, 400});
    auto dot_1 = std::make_shared<op::Dot>(skip_param_1_1, param1_2_reshape);

    auto bias1 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto broadcast_bias1 = std::make_shared<op::Broadcast>(bias1, Shape{10, 400}, AxisSet{0});
    auto add_1 = std::make_shared<op::Add>(dot_1, broadcast_bias1);

    // param2_1 -> xt (src_layer)
    auto param2_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 50});
    // param2_2 -> i2h weights (weights_layer)
    auto param2_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 50});
    auto param2_2_reshape =
        std::make_shared<op::Reshape>(param2_2, AxisVector{1, 0}, Shape{50, 400});
    auto dot_2 = std::make_shared<op::Dot>(param2_1, param2_2_reshape);
    auto bias2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto broadcast_bias2 = std::make_shared<op::Broadcast>(bias2, Shape{10, 400}, AxisSet{0});
    auto add_2 = std::make_shared<op::Add>(dot_2, broadcast_bias2);

    auto X = std::make_shared<op::Add>(add_2, add_1);
    // construct forget gate
    auto input_slice_0 = std::make_shared<op::Slice>(X, Coordinate{0, 0}, Coordinate{10, 100});
    auto forget_gate = std::make_shared<op::Sigmoid>(input_slice_0);

    // auto broadcast_pred = [](std::shared_ptr<Node> n) {
    //     return static_cast<bool>(std::dynamic_pointer_cast<op::Broadcast>(n));
    // };

    //ct-1 -> cell state (src_iter -> {ht | ct-1}
    auto ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    //auto skip_ct_1 = std::make_shared<pattern::op::Any>(ct_1, broadcast_pred);
    auto multiply_forget_gate_ct_1 = std::make_shared<op::Multiply>(forget_gate, ct_1);

    // construct input gate
    auto input_slice_1 = std::make_shared<op::Slice>(X, Coordinate{0, 100}, Coordinate{10, 200});
    auto input_gate = std::make_shared<op::Sigmoid>(input_slice_1);
    auto input_slice_2 = std::make_shared<op::Slice>(X, Coordinate{0, 200}, Coordinate{10, 300});
    auto tanh_1 = std::make_shared<op::Tanh>(input_slice_2);
    auto multiply_input_gate_tanh_1 = std::make_shared<op::Multiply>(input_gate, tanh_1);

    auto add_ct_1_input_gate_tanh_1 =
        std::make_shared<op::Add>(multiply_forget_gate_ct_1, multiply_input_gate_tanh_1);
    auto ct_label = std::make_shared<pattern::op::Label>(
        add_ct_1_input_gate_tanh_1, nullptr, NodeVector{add_ct_1_input_gate_tanh_1});

    // construct output gate
    auto input_slice_3 = std::make_shared<op::Slice>(X, Coordinate{0, 300}, Coordinate{10, 400});
    auto output_gate = std::make_shared<op::Sigmoid>(input_slice_3);
    auto tanh_2 = std::make_shared<op::Tanh>(ct_label);
    auto ht = std::make_shared<op::Multiply>(output_gate, tanh_2);
    auto ht_label = std::make_shared<pattern::op::Label>(ht, nullptr, NodeVector{ht});

    //Define a call back that needs to called once the DFG matches the pattern
    pattern::graph_rewrite_callback callback =
        [ct_label, param1_1, param1_2, param2_1, param2_2, bias1, bias2, ct_1](
            pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_fprop_lstm pattern against "
                         << m.match_root()->get_name();

            auto pattern_map = m.get_pattern_map();
            std::cout << "In Lstm fprop call back" << std::endl;

            // if (m.match_root()->get_element_type() != element::f32)
            // {
            //     NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name() << " type is not float!";
            //     return false;
            // }

            // if (m.match_root()->get_outputs().size() != pattern_map[input]->get_outputs().size())
            // {
            //     NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name()
            //                  << "input= " << pattern_map[input]->get_name() << "size dont match!";
            //     return false;
            // }

            //std::cout << "label_ct: " << join(label_ct[0]->get_shape()) <<  " " << label_ct[0]->get_name() << std::endl;
            Shape ct_shape{pattern_map[ct_label]->get_shape()};
            auto lstm = std::make_shared<op::Lstm>(pattern_map[param1_1],
                                                   pattern_map[param1_2],
                                                   pattern_map[param2_1],
                                                   pattern_map[param2_2],
                                                   pattern_map[bias1],
                                                   pattern_map[bias2],
                                                   pattern_map[ct_1],
                                                   ct_shape);

            auto ht_output = std::make_shared<op::GetOutputElement>(lstm, 0);
            auto ct_output = std::make_shared<op::GetOutputElement>(lstm, 1);

            std::vector<std::shared_ptr<Node>> new_args;
            for (auto node : pattern_map[ct_label]->get_users())
            {
                //std::cout << "Add_inputs: " << node->get_name() << std::endl;
                if (std::dynamic_pointer_cast<op::Multiply>(node))
                {
                    std::cout << "node_name: " << node->get_name() << std::endl;
                    for (size_t i = 0; i < node->get_input_size(); i++)
                    {
                        if (node->get_argument(i) == pattern_map[ct_label])
                        {
                            new_args.push_back(ct_output);
                        }
                        else
                        {
                            new_args.push_back(node->get_argument(i));
                        }
                        std::cout << "Multiply_input's shape: " << join(new_args[i]->get_shape())
                                  << " " << new_args[i]->get_name() << std::endl;
                    }
                    auto new_ct_node = node->copy_with_new_args(new_args);
                    std::cout << "node: " << node->get_name() << " replaced with  "
                              << new_ct_node->get_name() << std::endl;
                    ;
                    ngraph::replace_node(node, new_ct_node);
                    new_args.clear();
                }
            }
            ngraph::replace_node(m.match_root(), ht_output);
            return true;
        };
    //std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m = std::make_shared<pattern::Matcher>(ht, callback);
    std::cout << "lstm: " << m << std::endl;
    this->add_matcher(m);
}

std::shared_ptr<ngraph::Node> ngraph::runtime::cpu::pass::RecurrentCPUFusion::compute_rnn_args(
    std::vector<std::shared_ptr<pattern::op::Label>>& rnn_labels,
    pattern::RecurrentMatcher& m,
    bool input_symbol)
{
    std::cout << "Inside compute arg " << rnn_labels.size() << std::endl;
    std::set<std::shared_ptr<Node>> unique_params;
    NodeVector concat_args;
    for (size_t i = 0; i < rnn_labels.size(); i++)
    {
        auto node_lables = m.get_bound_nodes_for_pattern(rnn_labels[i]);
        std::cout << "rnn_label: " << node_lables[0]->get_name() << " "
                  << join(node_lables[0]->get_shape()) << " ";
        for (size_t j = 0; j < node_lables.size(); j++)
        {
            if (!std::dynamic_pointer_cast<op::GetOutputElement>(node_lables[j]) && !input_symbol)
            {
                unique_params.insert(node_lables[j]);
            }
            if (input_symbol)
            {
                unique_params.insert(node_lables[j]);
            }
        }
    }
    // push the uniques params as the Rnn arguments
    if (!unique_params.empty())
    {
        for (auto& param : unique_params)
        {
            //concat all the bounded params
            std::cout << "concat_args: " << param->get_name() << " " << join(param->get_shape())
                      << std::endl;
            concat_args.push_back(param);
        }
        std::cout << "++++++++++++++++++++" << std::endl;
        if (concat_args.size() > 1)
        {
            // reverse the concat_args so we concat in order of 0th, 1st,2nd....t'th time slice
            std::reverse(concat_args.begin(), concat_args.end());
            return std::make_shared<op::Concat>(concat_args, 0);
        }
    }
    return concat_args[0];
}

static bool is_unreachable(std::shared_ptr<ngraph::Node> n)
{
    std::unordered_set<std::shared_ptr<ngraph::Node>> instances_seen;
    std::deque<std::shared_ptr<ngraph::Node>> stack;
    stack.push_front(n);

    while (stack.size() > 0)
    {
        std::shared_ptr<ngraph::Node> n = stack.front();
        if (instances_seen.count(n) == 0)
        {
            if (n->is_output())
            {
                return false;
            }
            instances_seen.insert(n);
        }
        stack.pop_front();
        for (auto arg : n->get_users())
        {
            if (instances_seen.count(arg) == 0)
            {
                stack.push_front(arg);
            }
        }
    }
    return true;
}

void ngraph::runtime::cpu::pass::RecurrentCPUFusion::construct_rnn_fprop()
{
    // auto lstm_pred = [](std::shared_ptr<Node> n) {
    //     return static_cast<bool>(std::dynamic_pointer_cast<op::Lstm>(n));
    // };
    // auto goe_pred = [](std::shared_ptr<Node> n) {
    //     return static_cast<bool>(std::dynamic_pointer_cast<op::GetOutputElement>(n));
    // };
    // auto lstm_label =
    //     std::make_shared<pattern::op::Label>(element::f32, Shape{ }, lstm_pred);
    // auto goe = std::make_shared<pattern::op::Label>(lstm_label, goe_pred, NodeVector{lstm_label});

    // auto broadcast_pred = [](std::shared_ptr<Node> n) {
    //     return static_cast<bool>(std::dynamic_pointer_cast<op::Broadcast>(n));
    // };
    auto rpattern_ht_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});
    //auto skip_ht_1 = std::make_shared<pattern::op::Any>(rpattern_ht_1, broadcast_pred);
    auto weights_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto xt = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 200});
    auto weights_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto bias1 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto bias2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});

    auto lstm = std::make_shared<op::Lstm>(
        xt, weights_i2h, rpattern_ht_1, weights_h2h, bias1, bias2, ct_1, Shape{32, 100});
    auto goe = std::make_shared<op::GetOutputElement>(lstm, 0);
    auto lstm_node_label = std::make_shared<pattern::op::Label>(goe, nullptr, NodeVector{goe});

    pattern::recurrent_graph_rewrite_callback callback =
        [lstm_node_label, rpattern_ht_1, weights_h2h, xt, weights_i2h, bias1, bias2, ct_1, this](
            pattern::RecurrentMatcher& m) {

            static int count = 0;
            // if (count++ > 0)
            // return false;
            std::cout << "|||||||| In recurrent fusion |||||||" << std::endl;
            std::cout << "Xt: " << m.get_bound_nodes_for_pattern(xt).size() << std::endl;
            std::cout << "weights_i2h: " << m.get_bound_nodes_for_pattern(weights_i2h).size()
                      << std::endl;
            std::cout << "rpattern_ht_1: " << m.get_bound_nodes_for_pattern(rpattern_ht_1).size()
                      << std::endl;
            std::cout << "weights_h2h: " << m.get_bound_nodes_for_pattern(weights_h2h).size()
                      << std::endl;
            std::cout << "bias1: " << m.get_bound_nodes_for_pattern(bias1).size() << std::endl;
            std::cout << "bias2: " << m.get_bound_nodes_for_pattern(bias2).size() << std::endl;
            std::cout << "ct_1: " << m.get_bound_nodes_for_pattern(ct_1).size() << std::endl;

            auto ht_1_label = m.get_bound_nodes_for_pattern(rpattern_ht_1);

            std::vector<std::shared_ptr<pattern::op::Label>> src_iter_labels{rpattern_ht_1, ct_1};
            auto src_iter = this->compute_rnn_args(src_iter_labels, m);

            std::vector<std::shared_ptr<pattern::op::Label>> weights_layer_labels{weights_i2h};
            auto weights_layer = this->compute_rnn_args(weights_layer_labels, m);

            std::vector<std::shared_ptr<pattern::op::Label>> weights_iter_labels{weights_h2h};
            auto weights_iter = this->compute_rnn_args(weights_iter_labels, m);

            std::vector<std::shared_ptr<pattern::op::Label>> src_layer_labels{xt};
            auto src_layer = this->compute_rnn_args(src_layer_labels, m, true);

            auto bias_i2h_label = m.get_bound_nodes_for_pattern(bias2);
            auto bias_h2h_label = m.get_bound_nodes_for_pattern(bias1);
            auto bias = std::make_shared<op::Add>(bias_i2h_label[0], bias_h2h_label[0]);

            auto num_of_lstm_matched = m.get_number_of_recurrent_matches();
            size_t num_gates_in_lstm = 4;
            // TODO: assert for batch_size, sequence length and num_of_lstm's fused
            size_t batch_size = src_layer->get_shape()[0] / num_of_lstm_matched;
            size_t sequence_len = num_of_lstm_matched;
            size_t feature_size = ht_1_label[0]->get_shape()[1];

            auto rnn = std::make_shared<op::Rnn>(src_layer,
                                                 src_iter,
                                                 weights_layer,
                                                 weights_iter,
                                                 bias,
                                                 num_of_lstm_matched,
                                                 num_gates_in_lstm,
                                                 sequence_len,
                                                 feature_size);

            std::cout << "src_layer: " << join(src_layer->get_shape()) << std::endl;
            std::cout << "src_iter: " << join(src_iter->get_shape()) << std::endl;
            std::cout << "weights_layer: " << join(weights_layer->get_shape()) << std::endl;
            std::cout << "weights_iter: " << join(weights_iter->get_shape()) << std::endl;
            std::cout << "bias: " << join(bias->get_shape()) << std::endl;

            std::vector<std::shared_ptr<op::Slice>> ht_slice_per_timestep;
            auto rnn_ht_out = std::make_shared<op::GetOutputElement>(rnn, 0);
            auto rnn_ct_out = std::make_shared<op::GetOutputElement>(rnn, 1);

            //slice the rnn ht's
            size_t start_index = 0;
            size_t end_index = batch_size;
            for (size_t i = 0; i < num_of_lstm_matched; i++)
            {
                ht_slice_per_timestep.push_back(std::make_shared<op::Slice>(
                    rnn_ht_out, Coordinate{start_index, 0}, Coordinate{end_index, feature_size}));
                start_index += batch_size;
                end_index += batch_size;
            }

            std::cout << "rnn_time_slice: " << ht_slice_per_timestep.size() << std::endl;

            // find the lstm's nodes captured in PM
            auto lstm_goes = m.get_bound_nodes_for_pattern(lstm_node_label);
            std::set<std::shared_ptr<ngraph::Node>> lstm_nodes;
            for (size_t i = 0; i < lstm_goes.size(); i++)
            {
                // lstm's will be the input to GOE's
                lstm_nodes.insert(lstm_goes[i]->get_arguments()[0]);
            }

            std::cout << " ##### done collecting all lstm users #########"
                      << "num lstm's" << lstm_nodes.size() << std::endl;
            // collect all the consumers of LSTM goe's (ht)
            std::set<std::shared_ptr<ngraph::Node>> lstm_goe0_user;
            std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> map_nodes_to_goe;
            std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> map_goe_to_lstm_slices;
            std::shared_ptr<Node> goe_0;
            size_t index = 0;
            for (auto& node : lstm_nodes)
            {
                // now get the GOE0 which is the first output of lstm (ht)
                for (auto& goes : node->get_outputs().at(0).get_inputs())
                {
                    auto goe_node =
                        std::dynamic_pointer_cast<op::GetOutputElement>(goes->get_node());
                    // first output node of lstm
                    if (goe_node->get_n() == 0)
                    {
                        goe_0 = goes->get_node();
                    }
                }

                for (auto goe0_user : goe_0->get_users())
                {
                    if (lstm_nodes.find(goe0_user) == lstm_nodes.end() &&
                        !is_unreachable(goe0_user))
                    {
                        lstm_goe0_user.insert(goe0_user);
                        map_goe_to_lstm_slices[goe0_user] = ht_slice_per_timestep[index];
                        std::cout << "goe0_user " << goe0_user->get_name() << " ";
                    }
                }
                index++;
            }

            std::cout << "++ done collecting all lstm users ++++ " << std::endl;
            //now go through the lstm consumers and replace them with the slice
            std::vector<std::shared_ptr<Node>> new_args;
            for (auto& node : lstm_goe0_user)
            {
                for (auto& node_args : node->get_arguments())
                {
                    //std::cout << "lstm_inputs: " << (node->get_argument(pos)->get_name() << " input_size " << node->get_arguments().size() << std::endl;
                    if (std::find(lstm_goes.begin(), lstm_goes.end(), node_args) != lstm_goes.end())
                    {
                        std::cout << "index: " << index << " args_shape "
                                  << join(node_args->get_shape())
                                  << "name: " << node_args->get_name() << std::endl;
                        new_args.push_back(map_goe_to_lstm_slices[node]);
                    }
                    else
                    {
                        std::cout << " args_shape " << join(node_args->get_shape())
                                  << "name: " << node_args->get_name() << std::endl;
                        new_args.push_back(node_args);
                    }
                }
                std::cout << "node bring replaced " << node->get_name();
                auto new_node = node->copy_with_new_args(new_args);
                ngraph::replace_node(node, new_node);
                std::cout << "node: " << node->get_name() << " replaced with  "
                          << new_node->get_name() << std::endl;
                new_args.clear();
            }

            // //auto rnn_layer = std::make_shared<op::GetOutputElement>(rnn, 0);
            // std::cout << "In Recurrent Rnn matcher " << std::endl;
            // std::cout << "xt: " << xt_label.size() << " " << xt_label[0]->get_name() << std::endl;
            // std::cout << "ht: " << ht_1_label.size() << " " << ht_1_label[0]->get_name()
            //           << std::endl;
            // std::cout << "cell_state,ct-1: " << cell_state.size() << " "
            //           << cell_state[0]->get_name() << std::endl;
            // std::cout << "weights_h2h: " << weights_h2h_label.size() << " "
            //           << weights_h2h_label[0]->get_name() << std::endl;
            // std::cout << "weights_i2h: " << weights_h2h_label.size() << " "
            //           << weights_i2h_label[0]->get_name() << std::endl;
            // std::cout << "bias1: " << bias1_label.size() << " " << bias1_label[0]->get_name()
            //           << std::endl;
            // std::cout << "bias2: " << bias2_label.size() << " " << bias2_label[0]->get_name()
            //           << std::endl;
            std::cout << "<<<<<<<<<<<< End recurrent fusion >>>>>>>>>>>>>" << std::endl;
            ngraph::replace_node(m.get_match_root(),
                                 ht_slice_per_timestep[ht_slice_per_timestep.size() - 1]);
            return true;

        };

    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m = std::make_shared<pattern::RecurrentMatcher>(
        lstm_node_label, rpattern_ht_1, empty_correlated_matches, callback);
    this->add_matcher(m);
}
