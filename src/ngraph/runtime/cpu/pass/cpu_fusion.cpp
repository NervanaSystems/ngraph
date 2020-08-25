//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_set>

#include "cpu_fusion.hpp"
#include "ngraph/builder/make_constant.hpp"

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/conv_fused.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/gelu.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/dropout.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/quantized_matmul.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/util.hpp"

static bool init_cblas_arg(std::shared_ptr<ngraph::Node> reshape,
                           ngraph::Output<ngraph::Node> arg,
                           bool& transpose_w,
                           ngraph::Shape& shape_w)
{
    auto r_w = ngraph::as_type_ptr<ngraph::op::v0::Reshape>(reshape);

    if (!r_w)
    {
        if (arg.get_shape().size() != 2)
        {
            NGRAPH_DEBUG << arg << " 's rank != 2 " << ngraph::vector_to_string(arg.get_shape());
            return false;
        }
        return true; // nth to do; reshape isn't a reshape
    }

    if (r_w->get_output_shape(0).size() != 2)
    {
        NGRAPH_DEBUG << "Reshape for " << reshape->get_name() << " doesn't reshape into matrix"
                     << ngraph::vector_to_string(r_w->get_output_shape(0));
        return false;
    }

    auto io = r_w->get_input_order();
    if (r_w->get_output_shape(0).size() != arg.get_shape().size()) // reshape
    {
        auto dio = ngraph::get_default_order(io);
        if (io != dio) // we can't reshape and transpose at the same time
        {
            NGRAPH_DEBUG << "Reshape for " << reshape->get_name() << " is not in default order "
                         << ngraph::vector_to_string(io);
            NGRAPH_DEBUG << "r_w shape = " << ngraph::vector_to_string(r_w->get_output_shape(0));
            NGRAPH_DEBUG << "arg shape = " << ngraph::vector_to_string(arg.get_shape());
            return false;
        }

        shape_w = r_w->get_output_shape(0);
    }
    else
    {
        if (io == ngraph::AxisVector{1, 0})
        {
            transpose_w = true;
        }
        // otherwise no-op reshape
    }

    return true;
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_matmulbias()
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto W = std::make_shared<pattern::op::Label>(element::f32, shape_w);
    auto x = std::make_shared<pattern::op::Label>(element::f32, shape_x);
    auto b = std::make_shared<pattern::op::Label>(element::f32, shape_b);

    auto pmmb = std::make_shared<ngraph::op::MatmulBias>(
        W, x, Output<Node>(), W->get_output_shape(0), x->get_output_shape(0), false, false);
    auto pbroadcast =
        std::make_shared<ngraph::op::v0::Broadcast>(b, pmmb->get_output_shape(0), AxisSet{0});
    auto padd = pmmb + pbroadcast;

    auto callback = [W, x](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_matmulbias_pattern against node = "
                     << m.get_match_root()->get_name();

        auto mpattern = m.get_match_root(); // add
        auto m_matmul = ngraph::pattern::Matcher::unique_match<ngraph::op::MatmulBias>(mpattern);
        auto m_broadcast =
            ngraph::pattern::Matcher::unique_match<ngraph::op::v0::Broadcast>(mpattern);
        auto m_bias = m_broadcast->input_value(0);
        auto pvm = m.get_pattern_value_map();

        NGRAPH_CHECK(mpattern->get_output_element_type(0) != element::f64,
                     "Bias in DP MatMulBias is not supported yet");

        auto mmb = std::make_shared<ngraph::op::MatmulBias>(pvm[W],
                                                            pvm[x],
                                                            m_bias,
                                                            m_matmul->get_a_shape(),
                                                            m_matmul->get_b_shape(),
                                                            m_matmul->get_is_a_transposed(),
                                                            m_matmul->get_is_b_transposed(),
                                                            m_broadcast->get_broadcast_axes());

        m.get_match_value().replace(mmb->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(padd, "CPUFusion.MatMulBias");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_matmul()
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    Shape shape_dot{2, 1};

    auto W = std::make_shared<pattern::op::Label>(element::f32, shape_w);
    auto x = std::make_shared<pattern::op::Label>(element::f32, shape_x);

    auto reshape_pred = pattern::has_class<ngraph::op::v0::Reshape>();

    auto skip_w = std::make_shared<pattern::op::Skip>(W, reshape_pred);
    auto skip_x = std::make_shared<pattern::op::Skip>(x, reshape_pred);

    auto pdot = std::make_shared<ngraph::op::v0::Dot>(skip_w, skip_x);

    auto callback = [W, x](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_matmul_pattern against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();

        auto mpattern = m.get_match_root();
        auto dot = m.get_match_root();
        auto element_type = mpattern->get_output_element_type(0);

        if (element_type != element::f32 && element_type != element::f64)
        {
            NGRAPH_DEBUG << "mpattern = " << mpattern->get_name()
                         << " type is not float or double!";
            return false;
        }

        if (dot->get_output_shape(0).size() != 2)
        {
            NGRAPH_DEBUG << "dot = " << dot->get_name() << " shape is not equal to 2!";
            return false;
        }

        if (shape_size(dot->get_output_shape(0)) == 0)
        {
            NGRAPH_DEBUG << "dot has a zero dimension";
            return false;
        }

        bool transpose_w = false;
        Shape shape_arg0{pvm[W].get_shape()};
        if (!init_cblas_arg(dot->get_argument(0), pattern_map[W], transpose_w, shape_arg0))
        {
            return false;
        }

        bool transpose_x = false;
        Shape shape_arg1{pvm[x].get_shape()};
        if (!init_cblas_arg(dot->get_argument(1), pattern_map[x], transpose_x, shape_arg1))
        {
            return false;
        }

        auto cg = std::make_shared<ngraph::op::MatmulBias>(
            pvm[W], pvm[x], Output<Node>(), shape_arg0, shape_arg1, transpose_w, transpose_x);

        ngraph::replace_node(mpattern, cg);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(pdot, "CPUFusion.MatMul");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_fprop_bn()
{
    // construct varaiance
    auto N = ngraph::op::v0::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<ngraph::op::v1::Multiply>(input, input);
    auto sum_input = std::make_shared<ngraph::op::v0::Sum>(input, AxisSet{0});
    auto square_sumed_input = std::make_shared<ngraph::op::v1::Multiply>(sum_input, sum_input);
    auto sum_squared_input = std::make_shared<ngraph::op::v0::Sum>(input_sq, AxisSet{0});
    auto avg_input_sum_sq = std::make_shared<ngraph::op::v1::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<ngraph::op::v1::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<ngraph::op::v1::Divide>(xmu, N);
    auto variance_label =
        std::make_shared<pattern::op::Label>(variance, nullptr, OutputVector{variance});
    auto variance_with_broadcast =
        std::make_shared<ngraph::op::v0::Broadcast>(variance_label, Shape{2, 3}, AxisSet{0});

    // construct mean
    auto sum_input1 = std::make_shared<ngraph::op::v0::Sum>(input, AxisSet{0});
    auto mean = std::make_shared<ngraph::op::v1::Divide>(sum_input1, N);
    auto mean_label = std::make_shared<pattern::op::Label>(mean, nullptr, OutputVector{mean});
    auto mean_with_broadcast =
        std::make_shared<ngraph::op::v0::Broadcast>(mean_label, Shape{2, 3}, AxisSet{0});
    auto input_diff_mean = std::make_shared<ngraph::op::v1::Subtract>(input, mean_with_broadcast);

    // Eps
    auto eps_label = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
    auto eps_with_broadcast =
        std::make_shared<ngraph::op::v0::Broadcast>(eps_label, Shape{2, 3}, AxisSet{0});

    auto add1 = std::make_shared<ngraph::op::v1::Add>(eps_with_broadcast, variance_with_broadcast);
    auto sqrt_variance_eps = std::make_shared<ngraph::op::v0::Sqrt>(add1);
    auto divide_mean_variance =
        std::make_shared<ngraph::op::v1::Divide>(input_diff_mean, sqrt_variance_eps);

    // Gamma
    auto gamma_label = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
    auto gamma_with_broadcast =
        std::make_shared<ngraph::op::v0::Broadcast>(gamma_label, Shape{2, 3}, AxisSet{0});
    auto multiply_gamma =
        std::make_shared<ngraph::op::v1::Multiply>(gamma_with_broadcast, divide_mean_variance);

    // Beta
    auto beta_label = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
    auto beta_with_broadcast =
        std::make_shared<ngraph::op::v0::Broadcast>(beta_label, Shape{2, 3}, AxisSet{0});

    auto add_beta = std::make_shared<ngraph::op::v1::Add>(beta_with_broadcast, multiply_gamma);
    // This completes fprop bn pattern

    // Define a call back that needs to called once the DFG matches the pattern
    auto callback = [variance_label, mean_label, input, eps_label, gamma_label, beta_label](
                        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_fprop_bn pattern against "
                     << m.get_match_root()->get_name();

        // TODO - add assert's based on the matched node
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();
        NGRAPH_DEBUG << "Input: " << pattern_map[input]->get_name() << " "
                     << pvm[input].get_shape().size();
        NGRAPH_DEBUG << "Variance: " << pattern_map[variance_label]->get_name() << " "
                     << pvm[variance_label].get_shape().size();
        NGRAPH_DEBUG << "Mean: " << pattern_map[mean_label]->get_name() << " "
                     << pvm[mean_label].get_shape().size();
        NGRAPH_DEBUG << "eps: " << pattern_map[eps_label]->get_name() << " "
                     << pvm[eps_label].get_shape().size();
        NGRAPH_DEBUG << "gamma: " << pattern_map[gamma_label]->get_name() << " "
                     << pvm[gamma_label].get_shape().size();
        NGRAPH_DEBUG << "beta: " << pattern_map[beta_label]->get_name() << " "
                     << pvm[beta_label].get_shape().size();

        Shape bn_output_shape{m.get_match_value().get_shape()};
        Shape m_bn_mean_shape{pvm[mean_label].get_shape()};
        Shape m_bn_variance_shape{pvm[variance_label].get_shape()};

        // get epsilon value
        auto eps_ptr = as_type_ptr<ngraph::op::v0::Constant>(pattern_map[eps_label]);
        if (!eps_ptr)
        {
            NGRAPH_DEBUG << "Eps must be a constant";
            return false;
        }
        double epsilon = *(reinterpret_cast<const double*>(eps_ptr->get_data_ptr()));
        auto bn_node = std::make_shared<ngraph::op::v0::BatchNormTraining>(
            epsilon, pvm[gamma_label], pvm[beta_label], pvm[input]);

        if (!dnnl_utils::can_use_dnnl_batchnorm_fprop(bn_node.get()))
        {
            return false;
        }

        m.get_match_value().replace(bn_node->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add_beta, "CPUFusion.FpropBN");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_bias()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto pbias = std::make_shared<pattern::op::Label>(element::f32, Shape{});

    auto pbroadcast =
        std::make_shared<ngraph::op::v0::Broadcast>(pbias, shape, AxisSet{0, 1, 2, 3});

    auto pconv1 = std::make_shared<ngraph::op::v0::Convolution>(data_batch,
                                                                filters,
                                                                Strides{1, 1},
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{1, 1});
    auto p_conv_bias = pbroadcast + pconv1;

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_conv_bias against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto conv_m = as_type_ptr<ngraph::op::v0::Convolution>(
            m.get_match_value().get_node()->get_argument(0));
        auto bcast_m =
            as_type_ptr<op::v0::Broadcast>(m.get_match_value().get_node()->get_argument(1));

        if (conv_m == nullptr)
        {
            conv_m = as_type_ptr<ngraph::op::v0::Convolution>(
                m.get_match_value().get_node()->get_argument(1));
            bcast_m =
                as_type_ptr<op::v0::Broadcast>(m.get_match_value().get_node()->get_argument(0));
        }

        if (!runtime::cpu::dnnl_utils::can_use_dnnl_conv<ngraph::op::v0::Convolution>(conv_m.get()))
        {
            NGRAPH_DEBUG << "Convolution not supported by DNNL";
            return false;
        }

        // Except for the 2nd axis (channel dimension), we should either be broadcasting
        // to it or the dimension size should be 1.
        auto bcast_axes = bcast_m->get_broadcast_axes();
        for (size_t i = 0; i < bcast_m->get_output_shape(0).size(); i++)
        {
            if (i != 1 && bcast_axes.find(i) == bcast_axes.end() &&
                bcast_m->get_output_shape(0)[i] != 1)
            {
                return false;
            }
        }

        auto bias = bcast_m->get_argument(0);
        auto bias_shape = bias->get_output_shape(0);
        if (bias_shape.size() > 1)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << "conv_bias bias shape != 1, requires reshape to match filter count.";
            auto order = ngraph::get_default_order(bias_shape);
            auto bias_reshape = std::make_shared<ngraph::op::v0::Reshape>(
                bias, order, Shape{conv_m->get_input_shape(1)[0]});
            auto conv_bias =
                std::make_shared<ngraph::op::v0::ConvolutionBias>(conv_m, bias_reshape);
            m.get_match_value().replace(conv_bias->output(0));
        }
        else
        {
            auto conv_bias = std::make_shared<ngraph::op::v0::ConvolutionBias>(conv_m, bias);
            m.get_match_value().replace(conv_bias->output(0));
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_conv_bias, "CPUFusion.ConvBias");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_bias_bprop()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto delta = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto conv_bprop_filter =
        std::make_shared<ngraph::op::v0::ConvolutionBackpropFilters>(data_batch,
                                                                     shape,
                                                                     delta,
                                                                     Strides{1, 1},
                                                                     Strides{1, 1},
                                                                     CoordinateDiff{0, 0},
                                                                     CoordinateDiff{0, 0},
                                                                     Strides{1, 1});

    auto callback = [data_batch, delta](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_conv_bias_bprop against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();
        auto conv_bprop = m.get_match_root_as<ngraph::op::v0::ConvolutionBackpropFilters>();
        NGRAPH_CHECK(conv_bprop,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v0::ConvolutionBackpropFilters`");

        if (conv_bprop->get_input_shape(0).size() == 4 &&
            conv_bprop->get_input_shape(1).size() == 4 &&
            conv_bprop->get_input_element_type(0) == element::f32)
        {
            for (auto delta_user : pattern_map[delta]->get_users())
            {
                if (is_type<ngraph::op::v0::Sum>(delta_user))
                {
                    auto bias = as_type_ptr<ngraph::op::v0::Sum>(delta_user);
                    auto bias_shape = bias->get_output_shape(0);
                    bool flag = false;
                    if (bias_shape.size() > 1)
                    {
                        NGRAPH_DEBUG
                            << "mpattern = " << m.get_match_root()->get_name()
                            << "conv_bias bias shape != 1, requires reshape to match filter count.";
                        auto order = ngraph::get_default_order(bias_shape);
                        auto bias_reshape = std::make_shared<ngraph::op::v0::Reshape>(
                            bias, order, Shape{conv_bprop->get_filters_shape()[0]});
                        bias_shape = bias_reshape->get_output_shape(0);
                        flag = true;
                    }
                    auto conv_bias_bprop =
                        std::make_shared<ngraph::op::v0::ConvolutionBiasBackpropFiltersBias>(
                            pvm[data_batch],
                            conv_bprop->get_filters_shape(),
                            bias_shape,
                            pvm[delta],
                            conv_bprop->get_window_movement_strides_forward(),
                            conv_bprop->get_window_dilation_strides_forward(),
                            conv_bprop->get_padding_below_forward(),
                            conv_bprop->get_padding_above_forward(),
                            conv_bprop->get_data_dilation_strides_forward());
                    auto out0 = conv_bias_bprop->output(0);
                    auto out1 = conv_bias_bprop->output(1);
                    NGRAPH_DEBUG << "Replacing " << m.get_match_root()->get_name()
                                 << "with ConvolutionBiasBackpropFiltersBias";
                    m.get_match_value().replace(out0);
                    NGRAPH_DEBUG << "Replacing bias and adding it as a second o/p of "
                                    "ConvolutionBiasBackpropFiltersBias";
                    if (flag)
                    {
                        auto out1_reshape = std::make_shared<ngraph::op::v0::Reshape>(
                            out1, AxisVector{0}, delta_user->get_output_shape(0));
                        ngraph::replace_node(delta_user, out1_reshape);
                    }
                    else
                    {
                        ngraph::replace_node(delta_user, {out1});
                    }
                    return true;
                }
            }
        }
        return false;
    };

    auto m =
        std::make_shared<ngraph::pattern::Matcher>(conv_bprop_filter, "CPUFusion.ConvBiasBprop");
    this->add_matcher(m, callback);
}

static bool switch_nodes(std::shared_ptr<ngraph::Node> node1,
                         std::shared_ptr<ngraph::Node> node2,
                         size_t source_input_index = 0)
{
    // check if node1 has only 1 argument, not sure how it will work with >1 args
    if (node1->inputs().size() > 1)
    {
        NGRAPH_DEBUG << "Cannot switch. More than 1 inputs to this node\n";
        return false;
    }
    if (node1->get_users().size() > 1)
    {
        NGRAPH_DEBUG << "Cannot switch. More than 1 user of this node\n";
        return false;
    }
    if (node1->outputs().size() > 1)
    {
        NGRAPH_DEBUG << "Cannot switch. More than 1 output of this node\n";
        return false;
    }
    if (node2->outputs().size() > 1)
    {
        NGRAPH_DEBUG << "Cannot switch. More than 1 output of this node\n";
        return false;
    }

    auto target_inputs = node2->get_output_target_inputs(0);
    // Remove the control_dependency, which shouldn't be there, but in case
    // Other control_dependencies will work out fine even after switch.
    node2->remove_control_dependency(node1);

    // actual switch happening after this
    auto arg = node1->get_argument(source_input_index);
    node2->input(0).replace_source_output(arg);

    node1->input(0).replace_source_output(node2->output(0));

    // used implementation ref from replace_node
    for (auto& input : target_inputs)
    {
        input.replace_source_output(node1->output(0));
    }
    return true;
}

void ngraph::runtime::cpu::pass::CPUPreFusion::construct_maxpool_relu_switch()
{
    auto input_shape = Shape{1, 2, 2, 2};
    auto input = std::make_shared<pattern::op::Label>(element::f32, input_shape);
    Shape window_shape{2, 2};
    auto max_pool = std::make_shared<ngraph::op::v0::MaxPool>(input, window_shape);
    auto prelu = std::make_shared<ngraph::op::v0::Relu>(max_pool);

    auto callback = [input](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_maxpool_relu_switch against node = "
                     << m.get_match_root()->get_name();

        return switch_nodes(m.get_match_value().get_node()->get_argument(0), m.get_match_root());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, "CPUPreFusion.MaxpoolReluSwitch");
    this->add_matcher(m, callback);
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
    auto bn = std::make_shared<ngraph::op::v0::BatchNormTraining>(eps, gamma, beta, input);
    auto prelu = std::make_shared<ngraph::op::v0::Relu>(bn->output(0));

    auto callback = [input, gamma, beta](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_batch_norm_relu against node = "
                     << m.get_match_root()->get_name();

        auto pvm = m.get_pattern_value_map();
        auto m_bn = std::static_pointer_cast<ngraph::op::v0::BatchNormTraining>(
            m.get_match_root()->get_input_node_shared_ptr(0));
        if (!dnnl_utils::can_use_dnnl_batchnorm_fprop(m_bn.get()))
        {
            return false;
        }
        if (m_bn->output(0).get_target_inputs().size() > 1)
        {
            NGRAPH_DEBUG << "Relu isn't the only user of BatchNorm's output";
            return false;
        }

        auto bn_relu = std::make_shared<ngraph::op::BatchNormTrainingRelu>(
            m_bn->get_eps_value(), pvm[gamma], pvm[beta], pvm[input]);

        m_bn->output(0).replace(bn_relu->output(0));
        m_bn->output(1).replace(bn_relu->output(1));
        m_bn->output(2).replace(bn_relu->output(2));

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, "CPUFusion.BatchNormRelu");
    this->add_matcher(m, callback);
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
    auto bn_pred = [](Output<Node> node) {
        return pattern::has_class<ngraph::op::v0::BatchNormInference>()(node) ||
               pattern::has_class<ngraph::op::v0::BatchNormTraining>()(node);
    };
    auto bn = std::make_shared<pattern::op::Any>(
        input, bn_pred, OutputVector{gamma, beta, input, mean, var});
    auto prelu = std::make_shared<ngraph::op::v0::Relu>(bn);

    auto callback = [input, mean, var, gamma, beta](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_batch_norm_relu against node = "
                     << m.get_match_root()->get_name();

        auto pvm = m.get_pattern_value_map();

        auto bn_match = m.get_match_root()->get_input_node_shared_ptr(0);
        if (bn_match->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Relu isn't the only user of BatchNorm's output";
            return false;
        }

        std::shared_ptr<Node> bn_relu;
        if (auto bn_inference = as_type_ptr<ngraph::op::v0::BatchNormInference>(bn_match))
        {
            if (!dnnl_utils::can_use_dnnl_batchnorm_fprop(bn_inference.get()))
            {
                return false;
            }
            bn_relu =
                std::make_shared<ngraph::op::BatchNormInferenceRelu>(bn_inference->get_eps_value(),
                                                                     pvm[gamma],
                                                                     pvm[beta],
                                                                     pvm[input],
                                                                     pvm[mean],
                                                                     pvm[var]);
        }

        if (bn_relu)
        {
            m.get_match_value().replace(bn_relu->output(0));
            return true;
        }

        return false;
    };

    auto m =
        std::make_shared<ngraph::pattern::Matcher>(prelu, "CPUFusion.BatchNormReluGlobalStats");
    this->add_matcher(m, callback);
}

// graph before this fusion:
// input mean var gamma beta        broadcast1_input       broadcast2_input
//  \     \   |    /     /              /                        \
//       BatchNormInference          Broadcast1              Broadcast2
//             \                        /                        /
//                    Multiply                                 /
//                       \                                   /
//                                     Add
//                                      |
//                                     Relu
//
//
// graph after this fusion:
// input  mean var     gamma    broadcast1_input   beta     broadcast2_input
//  \      \    |        \          / \             /            /
//   \      \   |          Mulitply1      Multiply2             /
//    \      \  |             /              \                 /
//     \      \ |            /                    newAdd
//      \      \|           /                      /
//            BatchNormInferenceRelu
//
// Multiply1, Multiply2, and newAdd operate on vectors while Multiply an Add operate on
// multi-dimensional matrices.
// Multiply1, Multiply2, and newAdd may be folded away with constant folding pass later.
void ngraph::runtime::cpu::pass::CPUFusion::construct_batch_norm_infer_relu_with_multiply_add()
{
    auto input_shape = Shape{1, 3, 2, 2};
    auto input = std::make_shared<pattern::op::Label>(element::f32, input_shape);
    auto mean_shape = Shape{3};
    auto mean = std::make_shared<pattern::op::Label>(element::f32, mean_shape);
    auto var_shape = Shape{3};
    auto var = std::make_shared<pattern::op::Label>(element::f32, var_shape);
    auto gamma_shape = Shape{3};
    auto gamma = std::make_shared<pattern::op::Label>(element::f32, gamma_shape);
    auto beta_shape = Shape{3};
    auto beta = std::make_shared<pattern::op::Label>(element::f32, beta_shape);
    double eps = 0.001;
    auto bn =
        std::make_shared<ngraph::op::v0::BatchNormInference>(eps, gamma, beta, input, mean, var);
    auto bn_label = std::make_shared<pattern::op::Label>(bn, nullptr, OutputVector{bn});

    auto broadcast1_input = std::make_shared<pattern::op::Label>(element::f32, gamma_shape);
    auto broadcast1 = std::make_shared<ngraph::op::v0::Broadcast>(
        broadcast1_input, input_shape, AxisSet{0, 2, 3});
    auto broadcast1_label =
        std::make_shared<pattern::op::Label>(broadcast1, nullptr, OutputVector{broadcast1});
    auto multiply = std::make_shared<ngraph::op::v1::Multiply>(bn_label, broadcast1_label);
    auto multi_label =
        std::make_shared<pattern::op::Label>(multiply, nullptr, OutputVector{multiply});

    auto broadcast2_input = std::make_shared<pattern::op::Label>(element::f32, gamma_shape);
    auto broadcast2 = std::make_shared<ngraph::op::v0::Broadcast>(
        broadcast2_input, input_shape, AxisSet{0, 2, 3});
    auto broadcast2_label =
        std::make_shared<pattern::op::Label>(broadcast2, nullptr, OutputVector{broadcast2});
    auto add = std::make_shared<ngraph::op::v1::Add>(multi_label, broadcast2_label);
    auto prelu = std::make_shared<ngraph::op::v0::Relu>(add);

    auto callback = [input,
                     mean,
                     var,
                     gamma,
                     beta,
                     bn_label,
                     multi_label,
                     broadcast1_input,
                     broadcast2_input,
                     broadcast1_label,
                     broadcast2_label](pattern::Matcher& m) {
        NGRAPH_DEBUG
            << "In callback for construct_batch_norm_infer_relu_with_multi_add against node = "
            << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();

        auto bn_match = pattern_map[bn_label];
        if (bn_match->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Multiply isn't the only user of BatchNorm's output";
            return false;
        }
        auto multi_match = pattern_map[multi_label];
        if (multi_match->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Add isn't the only user of Multiply's output";
            return false;
        }

        std::vector<size_t> vec{0};
        for (size_t i = 2; i < pattern_map[input]->get_output_shape(0).size(); i++)
        {
            vec.push_back(i);
        }
        AxisSet axisSet{vec};
        if (std::static_pointer_cast<ngraph::op::v0::Broadcast>(pattern_map[broadcast1_label])
                    ->get_broadcast_axes() != axisSet ||
            std::static_pointer_cast<ngraph::op::v0::Broadcast>(pattern_map[broadcast2_label])
                    ->get_broadcast_axes() != axisSet)
        {
            NGRAPH_DEBUG << "Broadcast axes is not {0, 2, ...}";
            return false;
        }

        auto new_gamma =
            std::make_shared<ngraph::op::v1::Multiply>(pvm[gamma], pvm[broadcast1_input]);
        auto new_multi =
            std::make_shared<ngraph::op::v1::Multiply>(pvm[beta], pvm[broadcast1_input]);
        auto new_beta = std::make_shared<ngraph::op::v1::Add>(new_multi, pvm[broadcast2_input]);

        std::shared_ptr<Node> bn_relu;
        if (auto bn_inference = as_type_ptr<ngraph::op::v0::BatchNormInference>(bn_match))
        {
            if (!dnnl_utils::can_use_dnnl_batchnorm_fprop(bn_inference.get()))
            {
                return false;
            }
            bn_relu =
                std::make_shared<ngraph::op::BatchNormInferenceRelu>(bn_inference->get_eps_value(),
                                                                     new_gamma,
                                                                     new_beta,
                                                                     pvm[input],
                                                                     pvm[mean],
                                                                     pvm[var]);
        }

        if (bn_relu)
        {
            m.get_match_value().replace(bn_relu->output(0));
            return true;
        }

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu,
                                                        "CPUFusion.BatchNormInferReluWithMultiAdd");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_relu()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);

    auto pconv = std::make_shared<ngraph::op::v0::Convolution>(data_batch,
                                                               filters,
                                                               Strides{1, 1},
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});

    auto prelu = std::make_shared<ngraph::op::v0::Relu>(pconv);

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_relu against "
                     << m.get_match_root()->get_name();

        auto conv = std::static_pointer_cast<ngraph::op::v0::Convolution>(
            m.get_match_value().get_node()->get_argument(0));

        if (!runtime::cpu::dnnl_utils::can_use_dnnl_conv<ngraph::op::v0::Convolution>(conv.get()))
        {
            NGRAPH_DEBUG << "Convolution not supported by DNNL";
            return false;
        }

        if (conv->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Convolution has more than one user";
            return false;
        }

        auto conv_relu = std::make_shared<ngraph::op::ConvolutionRelu>(conv);
        m.get_match_value().replace(conv_relu->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(prelu, "CPUFusion.ConvRelu");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_bias_relu()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{shape[0]});

    auto conv_bias = std::make_shared<ngraph::op::v0::ConvolutionBias>(data_batch,
                                                                       filters,
                                                                       bias,
                                                                       Strides{1, 1},
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1});

    auto prelu = std::make_shared<ngraph::op::v0::Relu>(conv_bias);

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_relu against "
                     << m.get_match_root()->get_name();

        auto conv = std::static_pointer_cast<ngraph::op::v0::ConvolutionBias>(
            m.get_match_value().get_node()->get_argument(0));

        if (conv->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Convolution has more than one user";
            return false;
        }

        // ConvolutionBias created only if it can run with DNNL.
        // No further checks needed.
        auto conv_relu =
            std::make_shared<ngraph::op::v0::ConvolutionBias>(conv->input_value(0),
                                                              conv->input_value(1),
                                                              conv->input_value(2),
                                                              conv->get_window_movement_strides(),
                                                              conv->get_window_dilation_strides(),
                                                              conv->get_padding_below(),
                                                              conv->get_padding_above(),
                                                              conv->get_data_dilation_strides(),
                                                              true);
        m.get_match_value().replace(conv_relu->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(prelu, "CPUFusion.ConvBiasRelu");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_add()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);

    auto pconv = std::make_shared<ngraph::op::v0::Convolution>(data_batch,
                                                               filters,
                                                               Strides{1, 1},
                                                               Strides{1, 1},
                                                               CoordinateDiff{0, 0},
                                                               CoordinateDiff{0, 0},
                                                               Strides{1, 1});
    auto add_input = std::make_shared<pattern::op::Label>(element::f32, pconv->get_output_shape(0));
    auto padd = std::make_shared<ngraph::op::v1::Add>(add_input, pconv);

    auto callback = [data_batch, filters](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_add against "
                     << m.get_match_root()->get_name();

        auto add_m = m.get_match_root();
        auto conv_m = as_type_ptr<ngraph::op::v0::Convolution>(add_m->get_argument(1));
        auto inplace_input = add_m->get_argument(0);

        if (!conv_m)
        {
            conv_m = as_type_ptr<ngraph::op::v0::Convolution>(add_m->get_argument(0));
            inplace_input = add_m->get_argument(1);
        }

        if (!runtime::cpu::dnnl_utils::can_use_dnnl_conv<ngraph::op::v0::Convolution>(conv_m.get()))
        {
            NGRAPH_DEBUG << "Convolution not supported by DNNL";
            return false;
        }

        if (get_user_count(conv_m->output(0)) > 1)
        {
            NGRAPH_DEBUG << "Convolution has more than one user";
            return false;
        }

        if (inplace_input->is_parameter())
        {
            NGRAPH_DEBUG << "Skipping Convolution Add fusion due to parameter input";
            return false;
        }

        auto conv_add = std::make_shared<ngraph::op::ConvolutionAdd>(conv_m, inplace_input, false);
        m.get_match_value().replace(conv_add->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(padd, "CPUFusion.ConvAdd");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_add_relu()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto add_input = std::make_shared<pattern::op::Label>(element::f32, shape);

    auto pconv = std::make_shared<ngraph::op::ConvolutionAdd>(data_batch,
                                                              filters,
                                                              add_input,
                                                              Strides{1, 1},
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1},
                                                              false);
    auto prelu = std::make_shared<ngraph::op::v0::Relu>(pconv);

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_add_relu against "
                     << m.get_match_root()->get_name();

        auto conv_m = std::static_pointer_cast<ngraph::op::ConvolutionAdd>(
            m.get_match_value().get_node()->get_argument(0));
        if (conv_m->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Convolution has more than one user";
            return false;
        }

        // ConvolutionAdd created only if it can run with DNNL.
        // No further checks needed.
        auto conv_n =
            std::make_shared<ngraph::op::ConvolutionAdd>(conv_m->input_value(0),
                                                         conv_m->input_value(1),
                                                         conv_m->input_value(2),
                                                         conv_m->get_window_movement_strides(),
                                                         conv_m->get_window_dilation_strides(),
                                                         conv_m->get_padding_below(),
                                                         conv_m->get_padding_above(),
                                                         conv_m->get_data_dilation_strides(),
                                                         true);
        m.get_match_value().replace(conv_n->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(prelu, "CPUFusion.ConvAddRelu");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_bias_add()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{shape[0]});

    auto pconv = std::make_shared<ngraph::op::v0::ConvolutionBias>(data_batch,
                                                                   filters,
                                                                   bias,
                                                                   Strides{1, 1},
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{0, 0},
                                                                   CoordinateDiff{0, 0},
                                                                   Strides{1, 1});
    auto add_input = std::make_shared<pattern::op::Label>(element::f32, pconv->get_output_shape(0));
    auto padd = std::make_shared<ngraph::op::v1::Add>(add_input, pconv);

    auto callback = [data_batch, filters](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_sum against "
                     << m.get_match_root()->get_name();

        auto add_m = m.get_match_root();
        auto conv_m = as_type_ptr<ngraph::op::v0::ConvolutionBias>(add_m->get_argument(1));
        auto inplace_input = add_m->get_argument(0);

        if (!conv_m)
        {
            conv_m = as_type_ptr<ngraph::op::v0::ConvolutionBias>(add_m->get_argument(0));
            inplace_input = add_m->get_argument(1);
        }

        if (!runtime::cpu::dnnl_utils::can_use_dnnl_conv<ngraph::op::v0::ConvolutionBias>(
                conv_m.get()))
        {
            NGRAPH_DEBUG << "Convolution not supported by DNNL";
            return false;
        }

        if (get_user_count(conv_m->output(0)) > 1)
        {
            NGRAPH_DEBUG << "Convolution has more than one user";
            return false;
        }

        if (inplace_input->is_parameter())
        {
            NGRAPH_DEBUG << "Skipping Convolution Add fusion due to parameter input";
            return false;
        }

        auto conv_add =
            std::make_shared<ngraph::op::v0::ConvolutionBiasAdd>(conv_m, inplace_input, false);
        m.get_match_value().replace(conv_add->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(padd, "CPUFusion.ConvBiasAdd");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_dropout()
{
    Shape shape{1, 1, 2, 2};
    auto x = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto x_label = std::make_shared<pattern::op::Label>(x, nullptr, OutputVector{x});

    uint64_t seed = 1234;
    auto seed_label = std::make_shared<pattern::op::Label>(element::u64, Shape{0});

    double value = 0.9;
    auto value_const = ngraph::op::v0::Constant::create(element::f32, Shape{1, 1, 2, 2}, {value});
    auto value_label = std::make_shared<pattern::op::Label>(value_const);

    auto const1 = ngraph::op::v0::Constant::create(x->get_output_element_type(0), Shape{}, {1});
    auto const1_label = std::make_shared<pattern::op::Label>(const1);

    bool use_seed = false;
    auto use_seed_const = ngraph::op::v0::Constant::create(element::i32, Shape{}, {use_seed});
    auto use_seed_label = std::make_shared<pattern::op::Label>(use_seed_const);

    auto genmask = std::make_shared<op::v0::GenerateMask>(
        const1_label, x->get_output_shape(0), x->get_output_element_type(0), seed, value, use_seed);
    auto genmask_label =
        std::make_shared<pattern::op::Label>(genmask, nullptr, OutputVector{genmask});

    auto mult = std::make_shared<ngraph::op::v1::Multiply>(genmask_label, x_label);

    auto pdivide = std::make_shared<ngraph::op::v1::Divide>(mult, value_label);

    auto callback = [x, const1_label, seed_label, value_label, genmask_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_dropout against "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();

        auto m_div = m.get_match_root_as<ngraph::op::v1::Divide>();
        NGRAPH_CHECK(m_div,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v1::Divide`");

        auto gm =
            std::static_pointer_cast<ngraph::op::v0::GenerateMask>(pattern_map[genmask_label]);

        if (!is_type<ngraph::op::v0::Constant>(gm->get_argument(0)))
        {
            NGRAPH_DEBUG << "training argument to GenerateMask must be constant";
            return false;
        }
        if (!is_type<ngraph::op::v0::Constant>(gm->get_argument(2)))
        {
            NGRAPH_DEBUG << "use_seed argument to GenerateMask must be constant";
            return false;
        }
        if (!is_type<ngraph::op::v0::Constant>(gm->get_argument(3)))
        {
            NGRAPH_DEBUG << "seed argument to GenerateMask must be constant";
            return false;
        }
        if (!is_type<ngraph::op::v0::Constant>(gm->get_argument(4)))
        {
            NGRAPH_DEBUG << "probability argument to GenerateMask must be constant";
            return false;
        }

        auto dropout_n = std::make_shared<ngraph::op::Dropout>(
            pvm[x], gm->input_value(0), gm->input_value(2), gm->input_value(3), gm->input_value(4));

        m.get_match_value().replace(dropout_n->output(0));
        pvm[genmask_label].replace(dropout_n->output(1));

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(pdivide, "CPUFusion.Dropout");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_bias_add_relu()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{shape[0]});
    auto add_input = std::make_shared<pattern::op::Label>(element::f32, shape);

    auto pconv = std::make_shared<ngraph::op::v0::ConvolutionBiasAdd>(data_batch,
                                                                      filters,
                                                                      bias,
                                                                      add_input,
                                                                      Strides{1, 1},
                                                                      Strides{1, 1},
                                                                      CoordinateDiff{0, 0},
                                                                      CoordinateDiff{0, 0},
                                                                      Strides{1, 1},
                                                                      false);
    auto prelu = std::make_shared<ngraph::op::v0::Relu>(pconv);

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_sum against "
                     << m.get_match_root()->get_name();

        auto conv_m = std::static_pointer_cast<ngraph::op::v0::ConvolutionBiasAdd>(
            m.get_match_value().get_node()->get_argument(0));
        if (conv_m->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Convolution has more than one user";
            return false;
        }

        for (auto conv_bias_user : m.get_match_root()->get_users())
        {
            if (conv_bias_user->is_output())
            {
                // TODO: Remove restriction once we handle this case in codegen
                NGRAPH_DEBUG << "Unsafe to use in-place kernel since in-place output is a result";
                return false;
            }
        }

        // ConvolutionBiasAdd created only if it can run with DNNL.
        // No further checks needed.
        auto conv_n = std::make_shared<ngraph::op::v0::ConvolutionBiasAdd>(
            conv_m->input_value(0),
            conv_m->input_value(1),
            conv_m->input_value(2),
            conv_m->input_value(3),
            conv_m->get_window_movement_strides(),
            conv_m->get_window_dilation_strides(),
            conv_m->get_padding_below(),
            conv_m->get_padding_above(),
            conv_m->get_data_dilation_strides(),
            true);
        m.get_match_value().replace(conv_n->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(prelu, "CPUFusion.ConvBiasAddRelu");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_sigmoid_multiply()
{
    // Construct predicate to match sigmoid and tanh
    auto sigmoid_pred = [](Output<Node> n) {
        return (is_type<ngraph::op::v0::Sigmoid>(n.get_node())) ||
               (is_type<ngraph::op::v0::Tanh>(n.get_node()));
    };
    // Construct predicate to match other valid nodes
    auto other_pred = [](Output<Node> n) {
        return (is_type<ngraph::op::v0::Sigmoid>(n.get_node())) ||
               (is_type<ngraph::op::v0::Tanh>(n.get_node())) ||
               (is_type<ngraph::op::v1::Add>(n.get_node())) ||
               (is_type<ngraph::op::v0::Broadcast>(n.get_node()));
    };
    auto sigmoid_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1}, sigmoid_pred);
    auto sigmoid_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1}, other_pred);
    auto elem_mul = std::make_shared<ngraph::op::v1::Multiply>(sigmoid_0, sigmoid_1);

    auto callback = [sigmoid_0, sigmoid_1](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_sigmoid_multiply pattern against "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        if (m.get_match_value().get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << " type is not float!";
            return false;
        }

        using FunctionType = ngraph::op::SigmoidMultiply::FunctionType;
        const int max_inputs{2};
        std::array<std::shared_ptr<ngraph::Node>, max_inputs> match_nodes{
            {pattern_map[sigmoid_0], pattern_map[sigmoid_1]}};
        std::array<std::shared_ptr<ngraph::Node>, max_inputs> input_nodes;
        std::array<FunctionType, max_inputs> input_type;
        for (int i = 0; i < max_inputs; ++i)
        {
            input_type[i] = ngraph::op::SigmoidMultiply::identify_node_type(match_nodes[i]);
            if (input_type[i] != FunctionType::Identity)
            {
                if (match_nodes[i]->get_users().size() > 1)
                {
                    NGRAPH_DEBUG << "input node has multiple users, skipping fusion.";
                    return false;
                }
                input_nodes[i] = match_nodes[i]->get_argument(0);
            }
            else
            {
                input_nodes[i] = match_nodes[i];
            }
        }
        auto sigmoid_mul_node = std::make_shared<ngraph::op::SigmoidMultiply>(
            input_nodes[0], input_nodes[1], input_type[0], input_type[1]);
        m.get_match_value().replace(sigmoid_mul_node->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(elem_mul, "CPUFusion.SigmoidMultiply");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_leaky_relu()
{
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto iconst1 = ngraph::op::v0::Constant::create(element::f32, Shape{}, {1});
    auto alpha = std::make_shared<pattern::op::Label>(iconst1);
    auto broadcast_pred = [](Output<Node> n) {
        return (is_type<ngraph::op::v0::Broadcast>(n.get_node()));
    };
    auto skip_broadcast = std::make_shared<pattern::op::Skip>(alpha, broadcast_pred);
    auto leaky_relu = std::make_shared<ngraph::op::v1::Maximum>(
        input, std::make_shared<ngraph::op::v1::Multiply>(input, skip_broadcast));

    auto callback = [input, alpha](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_leaky_relu against "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        if (!is_type<ngraph::op::v0::Constant>(pattern_map[alpha]))
        {
            NGRAPH_DEBUG << "alpha must be constant for leaky relu";
            return false;
        }

        if (pattern_map[alpha]->get_output_element_type(0) != element::f32)
        {
            NGRAPH_DEBUG << "Only float negative slope supported for leaky relu";
            return false;
        }

        auto alpha_const_op =
            std::static_pointer_cast<ngraph::op::v0::Constant>(pattern_map[alpha]);
        auto alpha_vec = alpha_const_op->get_vector<float>();
        for (auto val : alpha_vec)
        {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
            if (val != alpha_vec[0])
            {
                NGRAPH_DEBUG << "alpha is not a singular constant";
                return false;
            }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
        }

        if (alpha_vec[0] < 0)
        {
            NGRAPH_DEBUG << "alpha is not positive";
            return false;
        }

        auto cg = std::make_shared<ngraph::op::CPULeakyRelu>(pattern_map[input], alpha_vec[0]);
        m.get_match_value().replace(cg->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(leaky_relu, "CPUFusion.CPULeakyRelu");
    this->add_matcher(m, callback);
}
void ngraph::runtime::cpu::pass::CPUFusion::construct_bounded_relu()
{
    auto relu_input = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto relu = std::make_shared<ngraph::op::v0::Relu>(relu_input);
    auto iconst1 = ngraph::op::v0::Constant::create(element::f32, Shape{}, {1});
    auto alpha = std::make_shared<pattern::op::Label>(iconst1);
    auto broadcast_pred = [](Output<Node> n) {
        return (is_type<ngraph::op::v0::Broadcast>(n.get_node()));
    };
    auto skip_broadcast = std::make_shared<pattern::op::Skip>(alpha, broadcast_pred);
    auto min = std::make_shared<ngraph::op::v1::Minimum>(relu, skip_broadcast);

    auto callback = [relu_input, alpha](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_bounded_relu against "
                     << m.get_match_root()->get_name();

        if (m.get_match_value().get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << " type is not float!";
            return false;
        }
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();
        if (!is_type<ngraph::op::v0::Constant>(pattern_map[alpha]))
        {
            NGRAPH_DEBUG << "alpha must be constant for bounded relu";
            return false;
        }

        // we wont fuse if the alpha and the Relu output element type are not same
        if (pvm[alpha].get_element_type() != pvm[relu_input].get_element_type())
        {
            return false;
        }
        if (pvm[alpha].get_shape() != pvm[relu_input].get_shape())
        {
            return false;
        }

        auto alpha_const_op =
            std::static_pointer_cast<ngraph::op::v0::Constant>(pattern_map[alpha]);
        float alpha_val = *(static_cast<float const*>(alpha_const_op->get_data_ptr()));
        NGRAPH_DEBUG << "relu_input: " << pattern_map[relu_input] << " min_val: "
                     << *(static_cast<float const*>(alpha_const_op->get_data_ptr()));

        auto cg = std::make_shared<ngraph::op::BoundedRelu>(pattern_map[relu_input], alpha_val);
        m.get_match_value().replace(cg->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(min, "CPUFusion.BoundedRelu");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_bias_folded_batch_norm()
{
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 2, 1, 1});
    auto filters = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 2, 1, 1});
    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{2});

    auto pconv = std::make_shared<ngraph::op::v0::ConvolutionBias>(input,
                                                                   filters,
                                                                   bias,
                                                                   Strides{1, 1},
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{0, 0},
                                                                   CoordinateDiff{0, 0},
                                                                   Strides{1, 1});

    auto mean = std::make_shared<pattern::op::Label>(element::f32, Shape{2});
    auto var = std::make_shared<pattern::op::Label>(element::f32, Shape{2});
    auto gamma = std::make_shared<pattern::op::Label>(element::f32, Shape{2});
    auto beta = std::make_shared<pattern::op::Label>(element::f32, Shape{2});
    double eps = 0.001;
    auto bn =
        std::make_shared<ngraph::op::v0::BatchNormInference>(eps, gamma, beta, pconv, mean, var);

    auto callback = [input, filters, bias, mean, var, gamma, beta](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for folded batch norm against node = "
                     << m.get_match_root()->get_name();
        auto pvm = m.get_pattern_value_map();

        auto m_bn = m.get_match_root_as<ngraph::op::v0::BatchNormInference>();
        NGRAPH_CHECK(m_bn,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v0::BatchNormInference`");
        auto m_conv =
            std::static_pointer_cast<ngraph::op::v0::ConvolutionBias>(m_bn->get_argument(2));

        if (m_conv->get_users().size() > 1)
        {
            return false;
        }

        if (m_conv->get_output_shape(0).size() != 4)
        {
            return false;
        }

        // new weights = old weights * gamma / sqrt(variance + epsilon)
        // new biases = (old_bias-mean) * gamma / sqrt(variance + epsilon) + beta

        auto bn_eps =
            ngraph::op::v0::Constant::create(element::f32, Shape{}, {m_bn->get_eps_value()});
        auto var_eps = std::make_shared<ngraph::op::v1::Add>(
            pvm[var],
            std::make_shared<ngraph::op::v0::Broadcast>(bn_eps, pvm[var].get_shape(), AxisSet{0}));
        auto sqrt_var_eps = std::make_shared<ngraph::op::v0::Sqrt>(var_eps);

        auto mean_gamma = std::make_shared<ngraph::op::v1::Multiply>(
            std::make_shared<ngraph::op::v1::Subtract>(pvm[bias], pvm[mean]), pvm[gamma]);
        auto new_biases = std::make_shared<ngraph::op::v1::Add>(
            pvm[beta], std::make_shared<ngraph::op::v1::Divide>(mean_gamma, sqrt_var_eps));
        auto weight_scaling = std::make_shared<ngraph::op::v1::Divide>(pvm[gamma], sqrt_var_eps);
        auto new_weights = std::make_shared<ngraph::op::v1::Multiply>(
            pvm[filters],
            std::make_shared<ngraph::op::v0::Broadcast>(
                weight_scaling, pvm[filters].get_shape(), AxisSet{1, 2, 3}));

        auto conv_bias =
            std::make_shared<ngraph::op::v0::ConvolutionBias>(pvm[input],
                                                              new_weights,
                                                              new_biases,
                                                              m_conv->get_window_movement_strides(),
                                                              m_conv->get_window_dilation_strides(),
                                                              m_conv->get_padding_below(),
                                                              m_conv->get_padding_above(),
                                                              m_conv->get_data_dilation_strides());
        m.get_match_value().replace(conv_bias->output(0));

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(bn, "CPUFusion.ConvBiasFoldedBatchNorm");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_conv_bias_affine_folding()
{
    // A * ConvBias (input, filters, bias) -> ConvBias (input, filters * A_c)
    Shape shape{2, 2, 1, 1};
    auto input = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{2});

    auto conv = std::make_shared<ngraph::op::v0::ConvolutionBias>(input,
                                                                  filters,
                                                                  bias,
                                                                  Strides{1, 1},
                                                                  Strides{1, 1},
                                                                  CoordinateDiff{0, 0},
                                                                  CoordinateDiff{0, 0},
                                                                  Strides{1, 1});
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, OutputVector{conv});

    auto Ac = std::make_shared<pattern::op::Label>(element::f32, Shape{2});
    auto A = std::make_shared<ngraph::op::v0::Broadcast>(Ac, Shape{2, 2, 1, 1}, AxisSet{0, 2, 3});
    auto A_label = std::make_shared<pattern::op::Label>(A, nullptr, OutputVector{A});
    auto multiply = std::make_shared<ngraph::op::v1::Multiply>(conv_label, A_label);

    auto callback = [input, filters, bias, conv_label, A_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for conv affine folding against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();

        auto conv_m =
            std::static_pointer_cast<ngraph::op::v0::ConvolutionBias>(pattern_map[conv_label]);

        if (conv_m->get_users().size() > 1)
        {
            return false;
        }

        if (conv_m->get_output_shape(0).size() != 4)
        {
            return false;
        }

        if (conv_m->with_relu())
        {
            return false;
        }

        auto A_m = std::static_pointer_cast<ngraph::op::v0::Broadcast>(pattern_map[A_label]);

        // Check if values are being broadcast along channel (2nd) dimension
        auto is_channel_bcast = [](const std::shared_ptr<ngraph::op::v0::Broadcast>& bcast) {
            auto input_shape = bcast->get_input_shape(0);
            if (input_shape.size() == 0 || shape_size(input_shape) == 1)
            {
                return true;
            }

            if (input_shape.size() == 1 && bcast->get_broadcast_axes() == AxisSet{0, 2, 3})
            {
                return true;
            }

            if (input_shape.size() == 2)
            {
                if (input_shape[0] == 1 && bcast->get_broadcast_axes() == AxisSet{2, 3})
                    return true;
            }
            return false;
        };

        if (!is_channel_bcast(A_m))
        {
            return false;
        }

        auto get_bcast_input = [](const std::shared_ptr<ngraph::op::v0::Broadcast>& bcast) {
            auto input_shape = bcast->get_input_shape(0);
            if (input_shape.size() == 0)
            {
                Shape bshape{bcast->get_output_shape(0)[1]};
                return std::static_pointer_cast<ngraph::Node>(
                    std::make_shared<ngraph::op::v0::Broadcast>(
                        bcast->input_value(0), bshape, AxisSet{0}));
            }
            if (shape_size(input_shape) == 1)
            {
                Shape bshape{bcast->get_output_shape(0)[1]};
                return std::static_pointer_cast<ngraph::Node>(
                    std::make_shared<ngraph::op::v0::Broadcast>(
                        std::make_shared<ngraph::op::v0::Reshape>(
                            bcast->input_value(0), get_default_order(input_shape), Shape{}),
                        bshape,
                        AxisSet{0}));
            }
            if (input_shape.size() == 1)
            {
                return bcast->get_argument(0);
            }
            if (input_shape.size() == 2)
            {
                Shape bshape{input_shape[1]};
                return std::static_pointer_cast<ngraph::Node>(
                    std::make_shared<ngraph::op::v0::Reshape>(
                        bcast->get_argument(0), AxisVector{0, 1}, bshape));
            }
            throw ngraph_error("Unexpected shape for bcast input");
        };

        auto Ac_m = get_bcast_input(A_m);

        // new weights = old weights * Ac_m
        // new_bias = old_bias * Ac_m;

        auto filters_n = std::make_shared<ngraph::op::v1::Multiply>(
            pvm[filters],
            std::make_shared<ngraph::op::v0::Broadcast>(
                Ac_m, pvm[filters].get_shape(), AxisSet{1, 2, 3}));

        auto bias_n = std::make_shared<ngraph::op::v1::Multiply>(pvm[bias], Ac_m);

        auto convbias_n =
            std::make_shared<ngraph::op::v0::ConvolutionBias>(pvm[input],
                                                              filters_n,
                                                              bias_n,
                                                              conv_m->get_window_movement_strides(),
                                                              conv_m->get_window_dilation_strides(),
                                                              conv_m->get_padding_below(),
                                                              conv_m->get_padding_above(),
                                                              conv_m->get_data_dilation_strides());
        m.get_match_value().replace(convbias_n->output(0));

        return true;
    };

    auto m =
        std::make_shared<ngraph::pattern::Matcher>(multiply, "CPUFusion.ConvBiasAffineFolding");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_groupconv_batchnorm_global_stats_folding()
{
    Shape shape_a{1, 32, 2, 2};
    Shape shape_b{32, 1, 1, 1};
    Shape shape_r{1, 32, 2, 2};

    auto input = std::make_shared<pattern::op::Label>(element::f32, shape_a);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape_b);
    auto resShape = std::make_shared<pattern::op::Label>(element::f32, shape_r);

    auto conv = std::make_shared<ngraph::op::v0::GroupConvolution>(input,
                                                                   filters,
                                                                   Strides{1, 1},
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{0, 0},
                                                                   CoordinateDiff{0, 0},
                                                                   Strides{1, 1},
                                                                   32);
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, OutputVector{conv});

    auto mean = std::make_shared<pattern::op::Label>(element::f32, Shape{32});
    auto var = std::make_shared<pattern::op::Label>(element::f32, Shape{32});
    auto gamma = std::make_shared<pattern::op::Label>(element::f32, Shape{32});
    auto beta = std::make_shared<pattern::op::Label>(element::f32, Shape{32});
    double eps = 0.001;
    auto bn = std::make_shared<ngraph::op::v0::BatchNormInference>(
        eps, gamma, beta, conv_label, mean, var);

    auto callback = [input, filters, conv_label, mean, var, gamma, beta](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for groupconv BatchNorm folding against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();

        auto m_bn = m.get_match_root_as<ngraph::op::v0::BatchNormInference>();
        NGRAPH_CHECK(m_bn,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v0::BatchNormInference`");
        auto conv_m =
            std::static_pointer_cast<ngraph::op::v0::GroupConvolution>(pattern_map[conv_label]);

        if (conv_m->get_users().size() > 1)
        {
            return false;
        }

        if (conv_m->get_output_shape(0).size() != 4)
        {
            return false;
        }

        if (conv_m->get_groups() == 0)
        {
            return false;
        }

        if (conv_m->has_groups_in_filters())
        {
            return false;
        }

        // new weights = old weights * gamma / sqrt(variance + epsilon)
        // new biases = (-mean) * gamma / sqrt(variance + epsilon) + beta

        auto bn_eps =
            ngraph::op::v0::Constant::create(element::f32, Shape{}, {m_bn->get_eps_value()});

        auto var_eps = std::make_shared<ngraph::op::v1::Add>(
            pvm[var],
            std::make_shared<ngraph::op::v0::Broadcast>(bn_eps, pvm[var].get_shape(), AxisSet{0}));
        auto sqrt_var_eps = std::make_shared<ngraph::op::v0::Sqrt>(var_eps);

        auto weight_scaling = std::make_shared<ngraph::op::v1::Divide>(pvm[gamma], sqrt_var_eps);

        auto weight_scaling_bcast = std::make_shared<ngraph::op::v0::Broadcast>(
            weight_scaling, pvm[filters].get_shape(), AxisSet{1, 2, 3});

        auto new_weights =
            std::make_shared<ngraph::op::v1::Multiply>(pvm[filters], weight_scaling_bcast);
        auto mean_gamma = std::make_shared<ngraph::op::v1::Multiply>(pvm[mean], weight_scaling);
        auto new_biases = std::make_shared<ngraph::op::v1::Subtract>(pvm[beta], mean_gamma);

        auto g_conv_bias = std::make_shared<ngraph::op::GroupConvolutionBias>(
            pvm[input],
            new_weights,
            new_biases,
            conv_m->get_window_movement_strides(),
            conv_m->get_window_dilation_strides(),
            conv_m->get_padding_below(),
            conv_m->get_padding_above(),
            conv_m->get_data_dilation_strides(),
            conv_m->get_groups(),
            conv_m->get_output_shape(0),
            false,
            1.0);
        m.get_match_value().replace(g_conv_bias->output(0));

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(
        bn, "CPUFusion.GroupconvBatchNormGlobalStatsFolding");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::
    construct_groupconv_batchnorm_global_stats_folding_relu()
{
    Shape shape_a{1, 32, 2, 2};
    Shape shape_b{32, 1, 1, 1};
    Shape shape_r{1, 32, 2, 2};
    Shape shape_bias{32};
    Shape shape_num{0};

    auto input = std::make_shared<pattern::op::Label>(element::f32, shape_a);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape_b);
    auto bias = std::make_shared<pattern::op::Label>(element::f32, shape_bias);
    auto num = std::make_shared<pattern::op::Label>(element::f32, shape_num);

    auto conv = std::make_shared<ngraph::op::GroupConvolutionBias>(input,
                                                                   filters,
                                                                   bias,
                                                                   Strides{1, 1},
                                                                   Strides{1, 1},
                                                                   CoordinateDiff{0, 0},
                                                                   CoordinateDiff{0, 0},
                                                                   Strides{1, 1},
                                                                   32,
                                                                   shape_r,
                                                                   false,
                                                                   1.0);
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, OutputVector{conv});

    // GroupConv + BatchNorm + Relu -> GroupConvBias
    auto prelu = std::make_shared<ngraph::op::v0::Relu>(conv_label);

    auto callback = [input, filters, bias, num, conv_label, prelu](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for GroupConvBias + Relu folding against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto conv_m =
            std::static_pointer_cast<ngraph::op::GroupConvolutionBias>(pattern_map[conv_label]);
        auto relu_m = m.get_match_root_as<ngraph::op::v0::Relu>();
        NGRAPH_CHECK(
            relu_m, "match root node ", *m.get_match_root(), " not of type `ngraph::op::v0::Relu`");

        auto g_conv_bias_relu = std::make_shared<ngraph::op::GroupConvolutionBias>(
            conv_m->input_value(0),
            conv_m->input_value(1),
            conv_m->input_value(2),
            conv_m->get_window_movement_strides(),
            conv_m->get_window_dilation_strides(),
            conv_m->get_padding_below(),
            conv_m->get_padding_above(),
            conv_m->get_data_dilation_strides(),
            conv_m->get_groups(),
            conv_m->get_output_shape(0),
            true);
        m.get_match_value().replace(g_conv_bias_relu->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(
        prelu, "CPUFusion.GroupconvBatchNormGlobalStatsFoldingRelu");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_deconvolution_affine_folding()
{
    Shape data_batch_shape{100, 512, 4, 4};
    Shape filters_shape{64, 512, 4, 4};
    auto data_label = std::make_shared<pattern::op::Label>(element::f32, data_batch_shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, filters_shape);
    Shape conv_out_shape{100, 64, 1, 1};
    auto out_delta = std::make_shared<pattern::op::Label>(element::f32, conv_out_shape);

    auto conv = std::make_shared<op::v0::ConvolutionBackpropData>(data_label->get_output_shape(0),
                                                                  filters,
                                                                  out_delta,
                                                                  Strides{1, 1},
                                                                  Strides{1, 1},
                                                                  CoordinateDiff{0, 0},
                                                                  CoordinateDiff{0, 0},
                                                                  Strides{1, 1});
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, OutputVector{conv});

    auto mean = std::make_shared<pattern::op::Label>(element::f32, Shape{512});
    auto var = std::make_shared<pattern::op::Label>(element::f32, Shape{512});
    auto gamma = std::make_shared<pattern::op::Label>(element::f32, Shape{512});
    auto beta = std::make_shared<pattern::op::Label>(element::f32, Shape{512});
    double eps = 0.001;
    auto bn = std::make_shared<op::v0::BatchNormInference>(eps, gamma, beta, conv_label, mean, var);

    auto callback = [data_label, filters, out_delta, conv_label, mean, var, gamma, beta](
                        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for deconv affine folding against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();

        // Matcher guarantees this is the right type
        auto m_bn = m.get_match_root_as<op::v0::BatchNormInference>();
        NGRAPH_CHECK(m_bn,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `op::v0::BatchNormInference`");
        auto conv_m =
            std::static_pointer_cast<op::v0::ConvolutionBackpropData>(pattern_map[conv_label]);

        if (conv_m->get_users().size() > 1)
        {
            return false;
        }

        if (conv_m->get_output_shape(0).size() != 4)
        {
            return false;
        }

        // new weights = old weights * gamma / sqrt(variance + epsilon)
        // new biases = (-mean) * gamma / sqrt(variance + epsilon) + beta

        auto bn_eps = op::v0::Constant::create(element::f32, Shape{}, {m_bn->get_eps_value()});

        auto var_eps = std::make_shared<op::v1::Add>(
            pvm[var],
            std::make_shared<op::v0::Broadcast>(bn_eps, pvm[var].get_shape(), AxisSet{0}));
        auto sqrt_var_eps = std::make_shared<op::v0::Sqrt>(var_eps);

        auto weight_scaling = std::make_shared<op::v1::Divide>(pvm[gamma], sqrt_var_eps);

        auto weight_scaling_bcast = std::make_shared<op::v0::Broadcast>(
            weight_scaling, pvm[filters].get_shape(), AxisSet{0, 2, 3});

        auto new_weights = std::make_shared<op::v1::Multiply>(pvm[filters], weight_scaling_bcast);
        auto mean_gamma = std::make_shared<op::v1::Multiply>(pvm[mean], weight_scaling);
        auto new_biases = std::make_shared<op::v1::Subtract>(pvm[beta], mean_gamma);

        // Weights are in i,o,h,w relative to deconvolution. Flip them to o,i,h,w
        auto new_weights_reshape =
            std::make_shared<op::v0::Reshape>(new_weights,
                                              AxisVector{1, 0, 2, 3},
                                              Shape{new_weights->get_output_shape(0).at(1),
                                                    new_weights->get_output_shape(0).at(0),
                                                    new_weights->get_output_shape(0).at(2),
                                                    new_weights->get_output_shape(0).at(3)});

        auto g_conv_bprop_data_bias =
            std::make_shared<op::DeconvolutionBias>(conv_m->get_data_batch_shape(),
                                                    new_weights_reshape,
                                                    pvm[out_delta],
                                                    new_biases,
                                                    conv_m->get_window_movement_strides_forward(),
                                                    conv_m->get_window_dilation_strides_forward(),
                                                    conv_m->get_padding_below_forward(),
                                                    conv_m->get_padding_above_forward(),
                                                    conv_m->get_data_dilation_strides_forward(),
                                                    false);
        m.get_match_value().replace(g_conv_bprop_data_bias->output(0));
        return true;
    };

    auto m =
        std::make_shared<ngraph::pattern::Matcher>(bn, "CPUFusion.deconvolution_affine_folding");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_deconvolution_affine_folding_relu()
{
    Shape data_batch_shape{100, 512, 4, 4};
    Shape filters_shape{512, 64, 4, 4}; // Note: the weights are in o,i,h,w
    auto data_label = std::make_shared<pattern::op::Label>(element::f32, data_batch_shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, filters_shape);
    Shape conv_out_shape{100, 64, 1, 1};
    auto out_delta = std::make_shared<pattern::op::Label>(element::f32, conv_out_shape);

    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{512});

    auto deconvb = std::make_shared<op::DeconvolutionBias>(data_label->get_output_shape(0),
                                                           filters,
                                                           out_delta,
                                                           bias,
                                                           Strides{1, 1},
                                                           Strides{1, 1},
                                                           CoordinateDiff{0, 0},
                                                           CoordinateDiff{0, 0},
                                                           Strides{1, 1},
                                                           false);
    auto deconvb_label =
        std::make_shared<pattern::op::Label>(deconvb, nullptr, OutputVector{deconvb});
    auto prelu = std::make_shared<op::v0::Relu>(deconvb_label);

    auto callback = [data_label, filters, out_delta, deconvb_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for deconvbias+relu against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto deconvb_m =
            std::static_pointer_cast<op::DeconvolutionBias>(pattern_map[deconvb_label]);

        if (deconvb_m->get_users().size() > 1)
        {
            return false;
        }

        auto g_deconvbias_relu = std::make_shared<op::DeconvolutionBias>(
            deconvb_m->get_data_batch_shape(),
            deconvb_m->input_value(0),
            deconvb_m->input_value(1),
            deconvb_m->input_value(2),
            deconvb_m->get_window_movement_strides_forward(),
            deconvb_m->get_window_dilation_strides_forward(),
            deconvb_m->get_padding_below_forward(),
            deconvb_m->get_padding_above_forward(),
            deconvb_m->get_data_dilation_strides_forward(),
            true);
        m.get_match_value().replace(g_deconvbias_relu->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(
        prelu, "CPUFusion.deconvolution_affine_folding_relu");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::CPUFusion::construct_update_slice()
{
    Shape shape_a{2, 32, 2};
    Shape shape_b{1, 32, 2};

    auto input = std::make_shared<pattern::op::Label>(element::f32, shape_a);
    auto slice =
        std::make_shared<ngraph::op::v0::Slice>(input, Coordinate{1, 0, 0}, Coordinate{2, 32, 2});
    auto slice_label = std::make_shared<pattern::op::Label>(slice, nullptr, OutputVector{slice});
    auto update_input = std::make_shared<pattern::op::Label>(element::f32, shape_b);
    auto update = std::make_shared<ngraph::op::v1::Add>(update_input, slice_label);
    auto replace_slice = std::make_shared<ngraph::op::v0::ReplaceSlice>(
        input, update, Coordinate{1, 0, 0}, Coordinate{2, 32, 2});

    auto callback = [input, update_input, slice_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for update_slice = " << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();
        auto slice_m = std::static_pointer_cast<ngraph::op::v0::Slice>(pattern_map[slice_label]);
        auto replace_m = m.get_match_root_as<ngraph::op::v0::ReplaceSlice>();
        NGRAPH_CHECK(replace_m,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v0::ReplaceSlice`");
        if (replace_m->get_lower_bounds() != slice_m->get_lower_bounds() ||
            replace_m->get_upper_bounds() != slice_m->get_upper_bounds() ||
            replace_m->get_strides() != slice_m->get_strides())
        {
            NGRAPH_DEBUG
                << "Update slice cannot be created, slice and replace_slice are not compatible";
            return false;
        }

        if (slice_m->get_users().size() > 1 || replace_m->get_argument(1)->get_users().size() > 1)
        {
            NGRAPH_DEBUG << "Update slice cannot be created, intermediate values required";
            return false;
        }

        auto update_slice = std::make_shared<ngraph::op::UpdateSlice>(pvm[input],
                                                                      pvm[update_input],
                                                                      replace_m->get_lower_bounds(),
                                                                      replace_m->get_upper_bounds(),
                                                                      replace_m->get_strides());
        m.get_match_value().replace(update_slice->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(replace_slice, "CPUFusion.UpdateSlice");
    this->add_matcher(m, callback);
}

// QuantizedConvolution + Dequantize + Relu -> QuantizedConvolutionRelu + Dequantize
void ngraph::runtime::cpu::pass::CPUQuantFusion::construct_qconv_relu(bool with_bias)
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::u8, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::i8, shape);
    auto input_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto filter_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto output_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto requant_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
    auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});
    auto dq_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto dq_zp = std::make_shared<pattern::op::Label>(element::i8, Shape{});

    std::shared_ptr<ngraph::op::Op> qconv;
    if (with_bias)
    {
        auto bias = std::make_shared<pattern::op::Label>(element::i32, Shape{shape[0]});
        qconv = std::make_shared<ngraph::op::v0::QuantizedConvolutionBias>(data_batch,
                                                                           filters,
                                                                           bias,
                                                                           Strides{1, 1},
                                                                           Strides{1, 1},
                                                                           CoordinateDiff{0, 0},
                                                                           CoordinateDiff{0, 0},
                                                                           Strides{1, 1},
                                                                           requant_scale,
                                                                           false);
    }
    else
    {
        qconv = std::make_shared<ngraph::op::v0::QuantizedConvolution>(data_batch,
                                                                       filters,
                                                                       Strides{1, 1},
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1},
                                                                       input_scale,
                                                                       uint8_zero,
                                                                       filter_scale,
                                                                       int8_zero,
                                                                       output_scale,
                                                                       int8_zero,
                                                                       element::i8,
                                                                       AxisSet{},
                                                                       AxisSet{},
                                                                       AxisSet{});
    }
    auto dq = std::make_shared<ngraph::op::v0::Dequantize>(
        qconv, dq_scale, dq_zp, element::f32, AxisSet{});
    auto relu = std::make_shared<ngraph::op::v0::Relu>(dq);

    auto callback = [with_bias](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_qconv_relu against "
                     << m.get_match_root()->get_name();

        auto dq_m = std::static_pointer_cast<ngraph::op::v0::Dequantize>(
            m.get_match_value().get_node()->get_argument(0));

        if (!(ngraph::is_zero(dq_m->get_argument(2))))
        {
            NGRAPH_DEBUG << "Non-zero zero point";
            return false;
        }

        if (dq_m->input_value(0).get_users().size() > 1)
        {
            NGRAPH_DEBUG << "QuantizedConvolutionBias has more than one user";
            return false;
        }

        if (!with_bias)
        {
            if (!runtime::cpu::dnnl_utils::can_use_dnnl_conv<ngraph::op::v0::QuantizedConvolution>(
                    dq_m->get_argument(0).get()))
            {
                NGRAPH_DEBUG << "Quantized Convolution not supported by DNNL";
                return false;
            }
        }

        std::shared_ptr<ngraph::op::Op> qconv_n;
        if (with_bias)
        {
            auto qconv_m = std::static_pointer_cast<ngraph::op::v0::QuantizedConvolutionBias>(
                dq_m->get_argument(0));
            qconv_n = std::make_shared<ngraph::op::v0::QuantizedConvolutionBias>(
                qconv_m->input_value(0),
                qconv_m->input_value(1),
                qconv_m->input_value(2),
                qconv_m->get_window_movement_strides(),
                qconv_m->get_window_dilation_strides(),
                qconv_m->get_padding_below(),
                qconv_m->get_padding_above(),
                qconv_m->get_data_dilation_strides(),
                qconv_m->input_value(3),
                true);
        }
        else
        {
            auto qconv_m = std::static_pointer_cast<ngraph::op::v0::QuantizedConvolution>(
                dq_m->get_argument(0));
            auto requantization_scale =
                qconv_m->get_argument(2) * qconv_m->get_argument(4) / qconv_m->get_argument(6);
            qconv_n = std::make_shared<ngraph::op::v0::QuantizedConvolutionRelu>(
                qconv_m->input_value(0),
                qconv_m->input_value(1),
                qconv_m->get_window_movement_strides(),
                qconv_m->get_window_dilation_strides(),
                qconv_m->get_padding_below(),
                qconv_m->get_padding_above(),
                qconv_m->get_data_dilation_strides(),
                requantization_scale);
        }
        auto zp = builder::make_constant<uint8_t>(element::u8, dq_m->get_input_shape(1), 0);
        auto dq_n = std::make_shared<ngraph::op::v0::Dequantize>(
            qconv_n, dq_m->input_value(1), zp, dq_m->get_output_element_type(0), dq_m->get_axes());
        m.get_match_value().replace(dq_n->output(0));
        return true;
    };

    std::shared_ptr<pattern::Matcher> m;
    if (with_bias)
    {
        m = std::make_shared<pattern::Matcher>(relu, "CPUQuantFusion.QConvBiasRelu");
    }
    else
    {
        m = std::make_shared<pattern::Matcher>(relu, "CPUQuantFusion.QConvRelu");
    }
    this->add_matcher(m, callback);
}

// Dequantize + AvgPool -> AvgPool + Dequantize
void ngraph::runtime::cpu::pass::CPUQuantFusion::construct_qavg_pool()
{
    Shape shape{2, 2, 1, 1};
    auto input = std::make_shared<pattern::op::Label>(element::i8, shape);
    auto dq_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto dq_zp = std::make_shared<pattern::op::Label>(element::i8, Shape{});
    auto dq = std::make_shared<ngraph::op::v0::Dequantize>(
        input, dq_scale, dq_zp, element::f32, AxisSet{});
    auto avg_pool = std::make_shared<ngraph::op::v0::AvgPool>(dq, Shape{1, 1});

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_qavg_pool against "
                     << m.get_match_root()->get_name();

        auto avg_pool_m = m.get_match_root_as<ngraph::op::v0::AvgPool>();
        NGRAPH_CHECK(avg_pool_m,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v0::AvgPool`");
        auto dq_m =
            std::static_pointer_cast<ngraph::op::v0::Dequantize>(avg_pool_m->get_argument(0));

        auto qavg_pool_n = std::make_shared<ngraph::op::v0::AvgPool>(
            dq_m->input_value(0),
            avg_pool_m->get_window_shape(),
            avg_pool_m->get_window_movement_strides(),
            avg_pool_m->get_padding_below(),
            avg_pool_m->get_padding_above(),
            avg_pool_m->get_include_padding_in_avg_computation());
        auto dq_n = std::make_shared<ngraph::op::v0::Dequantize>(qavg_pool_n,
                                                                 dq_m->input_value(1),
                                                                 dq_m->input_value(2),
                                                                 dq_m->get_output_element_type(0),
                                                                 dq_m->get_axes());
        m.get_match_value().replace(dq_n->output(0));
        return true;
    };

    this->add_matcher(std::make_shared<pattern::Matcher>(avg_pool, "CPUQuantFusion.QAvgPool"),
                      callback);
}

// Dequantize + Maxpool -> Maxpool + Dequantize
void ngraph::runtime::cpu::pass::CPUQuantFusion::construct_qmax_pool()
{
    Shape shape{2, 2, 1, 1};
    auto input = std::make_shared<pattern::op::Label>(element::i8, shape);
    auto dq_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto dq_zp = std::make_shared<pattern::op::Label>(element::i8, Shape{});
    auto dq = std::make_shared<ngraph::op::v0::Dequantize>(
        input, dq_scale, dq_zp, element::f32, AxisSet{});
    auto max_pool = std::make_shared<ngraph::op::v0::MaxPool>(dq, Shape{1, 1});

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_qmax_pool against "
                     << m.get_match_root()->get_name();

        auto max_pool_m = m.get_match_root_as<ngraph::op::v0::MaxPool>();
        NGRAPH_CHECK(max_pool_m,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v0::MaxPool`");
        auto dq_m =
            std::static_pointer_cast<ngraph::op::v0::Dequantize>(max_pool_m->get_argument(0));

        auto qmax_pool_n =
            std::make_shared<ngraph::op::v0::MaxPool>(dq_m->input_value(0),
                                                      max_pool_m->get_window_shape(),
                                                      max_pool_m->get_window_movement_strides(),
                                                      max_pool_m->get_padding_below(),
                                                      max_pool_m->get_padding_above());
        auto dq_n = std::make_shared<ngraph::op::v0::Dequantize>(qmax_pool_n,
                                                                 dq_m->input_value(1),
                                                                 dq_m->input_value(2),
                                                                 dq_m->get_output_element_type(0),
                                                                 dq_m->get_axes());
        m.get_match_value().replace(dq_n->output(0));
        return true;
    };

    this->add_matcher(std::make_shared<pattern::Matcher>(max_pool, "CPUQuantFusion.QMaxPool"),
                      callback);
}

// {Dequantize}* + Concat -> Concat + Dequantize
void ngraph::runtime::cpu::pass::CPUQuantFusion::construct_qconcat()
{
    Shape shape{2, 2, 1, 1};
    OutputVector inputs;
    NodeVector concats;
    // Pattern matcher looks for concats with exact number of inputs
    inputs.push_back(std::make_shared<pattern::op::Label>(element::f32, shape));
    // Concat2, Concat3, ... Concat6
    for (size_t i = 0; i < 5; i++)
    {
        inputs.push_back(std::make_shared<pattern::op::Label>(element::f32, shape));
        concats.push_back(std::make_shared<ngraph::op::v0::Concat>(inputs, 0));
    }

    auto callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_qconcat against "
                     << m.get_match_root()->get_name();

        auto concat_m = m.get_match_root_as<ngraph::op::v0::Concat>();
        NGRAPH_CHECK(concat_m,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v0::Concat`");
        auto dq_m = std::static_pointer_cast<ngraph::op::v0::Dequantize>(concat_m->get_argument(0));
        OutputVector new_args;
        for (auto arg : concat_m->get_arguments())
        {
            if (!is_type<op::v0::Dequantize>(arg))
            {
                return false;
            }

            // ensure dequant scales are same
            if (!ngraph::compare_constants(arg->get_argument(1), dq_m->get_argument(1)))
            {
                NGRAPH_DEBUG << "Concat: Dequantize scale must be same";
                return false;
            }

            new_args.push_back(arg->get_argument(0));
        }
        auto concat_n =
            std::make_shared<ngraph::op::v0::Concat>(new_args, concat_m->get_concatenation_axis());
        auto dq_n = std::make_shared<ngraph::op::v0::Dequantize>(concat_n,
                                                                 dq_m->input_value(1),
                                                                 dq_m->input_value(2),
                                                                 dq_m->get_output_element_type(0),
                                                                 dq_m->get_axes());
        m.get_match_value().replace(dq_n->output(0));

        return true;
    };

    for (size_t i = 0; i < 5; i++)
    {
        this->add_matcher(std::make_shared<pattern::Matcher>(
                              concats[i], "CPUQuantFusion.QConcat" + std::to_string(i + 2)),
                          callback);
    }
}

void ngraph::runtime::cpu::pass::CPUQuantFusion::construct_dq_q()
{
    Shape shape{2, 2, 1, 1};
    auto input = std::make_shared<pattern::op::Label>(element::i8, shape);
    auto dq_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto dq_zp = std::make_shared<pattern::op::Label>(element::i8, Shape{});

    auto q_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto q_zp = std::make_shared<pattern::op::Label>(element::i8, Shape{});

    auto dq = std::make_shared<ngraph::op::v0::Dequantize>(
        input, dq_scale, dq_zp, element::f32, AxisSet{});
    ngraph::op::v0::Quantize::RoundMode round_mode =
        ngraph::op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
    auto q = std::make_shared<ngraph::op::v0::Quantize>(
        dq, q_scale, q_zp, element::i8, AxisSet{}, round_mode);

    auto callback = [input](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_dq_q against "
                     << m.get_match_root()->get_name();

        auto q_m = m.get_match_root_as<ngraph::op::v0::Quantize>();
        NGRAPH_CHECK(q_m,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v0::Quantize`");
        auto dq_m = std::static_pointer_cast<ngraph::op::v0::Dequantize>(q_m->get_argument(0));
        if (!(ngraph::is_zero(q_m->get_argument(2)) && ngraph::is_zero(dq_m->get_argument(2))))
        {
            NGRAPH_DEBUG << "Non-zero zero points";
            return false;
        }

        if (m.get_match_value().get_element_type() !=
            m.get_pattern_value_map()[input].get_element_type())
        {
            NGRAPH_DEBUG << "Type mismatch between input and quantize output";
            return false;
        }

        if (!ngraph::compare_constants(q_m->get_argument(1), dq_m->get_argument(1)))
        {
            NGRAPH_DEBUG << "Scales dont match";
            return false;
        }

        m.get_match_value().replace(m.get_pattern_value_map()[input]);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(q, "CPUQuantFusion.DQandQ");
    this->add_matcher(m, callback);
}

// Left Branch(LB): QCONVB + DQ + {Reshape/Broadcast}
// Right Branch(RB): DQ + {Reshape/Broadcast}
// Relu(LB + RB) -> QCB{S}A
void ngraph::runtime::cpu::pass::CPUQuantFusion::construct_qconvb_add()
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::u8, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::i8, shape);
    auto bias = std::make_shared<pattern::op::Label>(element::i32, Shape{shape[1]});
    auto requantization_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto output_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto dq_scale1 = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto dq_zp1 = std::make_shared<pattern::op::Label>(element::i8, Shape{});
    auto dq_scale2 = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto dq_zp2 = std::make_shared<pattern::op::Label>(element::i8, Shape{});

    // Left Graph
    auto qconvb = std::make_shared<ngraph::op::v0::QuantizedConvolutionBias>(data_batch,
                                                                             filters,
                                                                             bias,
                                                                             Strides{1, 1},
                                                                             Strides{1, 1},
                                                                             CoordinateDiff{0, 0},
                                                                             CoordinateDiff{0, 0},
                                                                             Strides{1, 1},
                                                                             requantization_scale,
                                                                             false);
    auto qconvb_label = std::make_shared<pattern::op::Label>(qconvb, nullptr, OutputVector{qconvb});
    auto dq_l = std::make_shared<ngraph::op::v0::Dequantize>(
        qconvb_label, dq_scale1, dq_zp1, element::f32, AxisSet{});
    auto dq_l_label = std::make_shared<pattern::op::Label>(dq_l, nullptr, OutputVector{dq_l});
    auto skipr_l = std::make_shared<pattern::op::Skip>(
        dq_l_label, [](Output<Node> n) { return is_type<op::v0::Reshape>(n.get_node()); });
    auto skipb_l = std::make_shared<pattern::op::Skip>(
        skipr_l, [](Output<Node> n) { return is_type<op::v0::Broadcast>(n.get_node()); });

    // Right Graph
    auto summand = std::make_shared<pattern::op::Label>(element::i8, qconvb->get_output_shape(0));
    auto dq_r = std::make_shared<ngraph::op::v0::Dequantize>(
        summand, dq_scale2, dq_zp2, element::f32, AxisSet{});
    auto dq_r_label = std::make_shared<pattern::op::Label>(dq_r, nullptr, OutputVector{dq_r});
    auto skipr_r = std::make_shared<pattern::op::Skip>(
        dq_r_label, [](Output<Node> n) { return is_type<op::v0::Reshape>(n.get_node()); });
    auto skipb_r = std::make_shared<pattern::op::Skip>(
        skipr_r, [](Output<Node> n) { return is_type<op::v0::Broadcast>(n.get_node()); });

    // Add left + right
    auto add = skipb_l + skipb_r;
    auto prelu = std::make_shared<ngraph::op::v0::Relu>(add);

    auto callback = [dq_l_label, dq_r_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_qconvb_dq_add_relu against "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto add_m =
            as_type_ptr<ngraph::op::v1::Add>(m.get_match_value().get_node()->get_argument(0));
        auto dq_l_m = as_type_ptr<ngraph::op::v0::Dequantize>(pattern_map[dq_l_label]);
        auto dq_r_m = as_type_ptr<ngraph::op::v0::Dequantize>(pattern_map[dq_r_label]);

        // both left and right are QuantizedConvolutionBias
        if (is_type<op::v0::QuantizedConvolutionBias>(dq_r_m->get_argument(0)))
        {
            for (auto user : m.get_match_root()->get_users())
            {
                auto q_m = as_type_ptr<ngraph::op::v0::Quantize>(user);
                if (q_m)
                {
                    auto q_m_scale = q_m->get_argument(1);
                    auto dq_l_m_scale = dq_l_m->get_argument(1);
                    auto dq_r_m_scale = dq_r_m->get_argument(1);
                    if (!ngraph::compare_constants(q_m_scale, dq_l_m_scale) &&
                        ngraph::compare_constants(q_m_scale, dq_r_m_scale))
                    {
                        NGRAPH_DEBUG << "Scales of Q and DQ of right branch match";
                        // switch left and right branch
                        auto temp = dq_l_m;
                        dq_l_m = dq_r_m;
                        dq_r_m = temp;
                    }
                    break;
                }
            }
        }

        auto qconv = std::static_pointer_cast<ngraph::op::v0::QuantizedConvolutionBias>(
            dq_l_m->get_argument(0));
        auto inplace_input = dq_r_m->get_argument(0);

        if (!(ngraph::is_zero(dq_l_m->get_argument(2)) && ngraph::is_zero(dq_r_m->get_argument(2))))
        {
            NGRAPH_DEBUG << "Non-zero zero points";
            return false;
        }

        if (dq_r_m->get_input_element_type(0) != element::i8 &&
            dq_r_m->get_input_element_type(0) != element::u8)
        {
            NGRAPH_DEBUG << "Non int8/uint8 summand";
            return false;
        }

        if (get_user_count(qconv->output(0)) > 1)
        {
            NGRAPH_DEBUG << "QuantizedConvolutionBias has more than one user";
            return false;
        }

        // The next two checks are not required once we support fallbacks in dex/codegen
        // for non in-place input
        if (!is_post_dominated(inplace_input.get(), add_m.get()))
        {
            NGRAPH_DEBUG << "Unsafe to use in-place kernel since add's in-place input has "
                            "potential live users";
            return false;
        }

        if (inplace_input->is_parameter())
        {
            NGRAPH_DEBUG
                << "Unsafe to use in-place kernel since add's in-place input is a parameter";
            return false;
        }

        if (inplace_input->get_output_shape(0) != qconv->get_output_shape(0))
        {
            NGRAPH_DEBUG << "Summand shape doesn't match convolution shape";
            return false;
        }

        auto requant_scale = qconv->get_argument(3);
        auto dq_l_scale = dq_l_m->get_argument(1);
        auto dq_r_scale = dq_r_m->get_argument(1);
        auto sum_scale = (dq_r_scale / dq_l_scale);

        std::shared_ptr<ngraph::op::Op> qconvba;
        if (dq_r_m->get_input_element_type(2) == element::i8)
        {
            // TODO (jbobba): Investigate the need for Convert op
            qconvba = std::make_shared<ngraph::op::v0::Convert>(
                std::make_shared<ngraph::op::v0::QuantizedConvolutionBiasSignedAdd>(
                    qconv->input_value(0),
                    qconv->input_value(1),
                    qconv->input_value(2),
                    inplace_input,
                    qconv->get_window_movement_strides(),
                    qconv->get_window_dilation_strides(),
                    qconv->get_padding_below(),
                    qconv->get_padding_above(),
                    qconv->get_data_dilation_strides(),
                    requant_scale,
                    sum_scale,
                    true),
                element::u8);
        }
        else
        {
            qconvba = std::make_shared<ngraph::op::v0::QuantizedConvolutionBiasAdd>(
                qconv->input_value(0),
                qconv->input_value(1),
                qconv->input_value(2),
                inplace_input,
                qconv->get_window_movement_strides(),
                qconv->get_window_dilation_strides(),
                qconv->get_padding_below(),
                qconv->get_padding_above(),
                qconv->get_data_dilation_strides(),
                requant_scale,
                sum_scale,
                true);
        }
        auto zp = ngraph::op::v0::Constant::create(element::u8, Shape{}, {0});
        auto DQ = std::make_shared<ngraph::op::v0::Dequantize>(
            qconvba, dq_l_scale, zp, element::f32, AxisSet{});
        m.get_match_value().replace(DQ->output(0));

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(prelu, "CPUQuantFusion.QConvBiasSignedAdd");
    this->add_matcher(m, callback);
}

// Convert a QuantizedDot which takes [m,n]*[n,k] to
// QuantizedMatmul which reorders input1 and does [m,n]*[k,n]
// which is what dnnl wants
void ngraph::runtime::cpu::pass::CPUQuantFusion::construct_quantized_matmul()
{
    Shape shape_input0{2, 3};
    Shape shape_input1{3, 4};
    auto input0 = std::make_shared<pattern::op::Label>(element::u8, shape_input0);
    auto input1 = std::make_shared<pattern::op::Label>(element::i8, shape_input1);
    auto input0_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto input1_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto output_scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});

    auto int8_zero = op::v0::Constant::create(element::i8, Shape{}, {0});
    auto uint8_zero = op::v0::Constant::create(element::u8, Shape{}, {0});

    auto q_dot = std::make_shared<ngraph::op::v0::QuantizedDot>(input0,
                                                                input1,
                                                                1,
                                                                input0_scale,
                                                                uint8_zero,
                                                                input1_scale,
                                                                int8_zero,
                                                                output_scale,
                                                                int8_zero,
                                                                element::i8,
                                                                AxisSet{},
                                                                AxisSet{},
                                                                AxisSet{});
    auto callback = [input0, input1, input0_scale, input1_scale, output_scale](
                        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for Qdot against node = " << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();

        auto qdot = m.get_match_root_as<ngraph::op::v0::QuantizedDot>();
        NGRAPH_CHECK(qdot,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `ngraph::op::v0::QuantizedDot`");
        auto input_0 = pvm[input0];
        auto input_1 = pvm[input1];
        auto input_0_scale = pvm[input0_scale];
        auto input_1_scale = pvm[input1_scale];
        auto scale_output = pvm[output_scale];
        auto scale_new = input_0_scale * input_1_scale / scale_output;

        if (input_0.get_shape().size() != 2 || input_1.get_shape().size() != 2)
        {
            return false;
        }
        if (input_0.get_element_type() == element::u8 && input_1.get_element_type() == element::u8)
        {
            return false;
        }

        auto reshape_input1 = std::make_shared<op::v0::Reshape>(
            input_1, AxisVector{1, 0}, Shape{input_1.get_shape()[1], input_1.get_shape()[0]});
        auto qmatmul = std::make_shared<ngraph::op::QuantizedMatmul>(
            input_0, reshape_input1, scale_new, qdot->get_output_type());

        m.get_match_value().replace(qmatmul->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(q_dot, "CPUQuantFusion.QDot");
    this->add_matcher(m, callback);
}
