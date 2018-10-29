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

#include <algorithm>
#include <unordered_set>

#include "ngraph/pass/core_fusion.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"

using namespace ngraph;
using namespace std;

static shared_ptr<Node> construct_constant_node(int n)
{
    return op::Constant::create(element::f32, Shape{}, {n});
}

void pass::CoreFusion::construct_relu()
{
    auto iconst0 = construct_constant_node(0);
    auto val = make_shared<pattern::op::Label>(iconst0);
    auto zero = make_shared<pattern::op::Label>(iconst0, nullptr, NodeVector{iconst0});

    auto skip_broadcast =
        std::make_shared<pattern::op::Skip>(zero, pattern::has_class<op::Broadcast>());
    auto max = make_shared<op::Maximum>(skip_broadcast, val);

    pattern::graph_rewrite_callback callback = [val, zero](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_relu against "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        auto mzero = m.get_pattern_map()[zero];
        if (!ngraph::is_zero(mzero))
        {
            NGRAPH_DEBUG << "zero constant = " << mzero->get_name() << " not equal to 0\n";
            return false;
        }
        auto mpattern = m.get_match_root();

        auto cg = shared_ptr<Node>(new op::Relu(pattern_map[val]));
        ngraph::replace_node(m.get_match_root(), cg);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(max, callback);
    this->add_matcher(m);
}

void pass::CoreFusion::construct_sigmoid()
{
    // construct variance
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_input = std::make_shared<op::Negative>(input);
    auto exp_neg_input = std::make_shared<op::Exp>(neg_input);

    // broadcast input
    auto constant = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto broadcast_constant = std::make_shared<op::Broadcast>(constant, Shape{3, 4}, AxisSet{0, 1});

    auto add_exp = std::make_shared<op::Add>(exp_neg_input, broadcast_constant);
    auto divide_1_over_exp = std::make_shared<op::Divide>(broadcast_constant, add_exp);

    // Define a call back that needs to called once the DFG matches the pattern
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

void pass::CoreFusion::construct_sigmoid_bprop()
{
    // construct variance
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_input = std::make_shared<op::Negative>(input);
    auto exp_neg_input = std::make_shared<op::Exp>(neg_input);

    // broadcast input
    auto constant = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto broadcast_constant = std::make_shared<op::Broadcast>(constant, Shape{3, 4}, AxisSet{0, 1});

    auto add_exp = std::make_shared<op::Add>(exp_neg_input, broadcast_constant);
    // auto divide_1_over_exp = std::make_shared<op::Divide>(broadcast_constant, add_exp);
    auto sigmoid_fwd = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});

    auto delta = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_delta = std::make_shared<op::Negative>(delta);

    auto multiply_sigmoid_delta = std::make_shared<op::Multiply>(sigmoid_fwd, neg_delta);
    auto divide_2 = std::make_shared<op::Divide>(multiply_sigmoid_delta, add_exp);

    auto multiply_2 = std::make_shared<op::Multiply>(divide_2, exp_neg_input);
    auto negtive_2 = std::make_shared<op::Negative>(multiply_2);

    // Define a call back that needs to called once the DFG matches the pattern
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

void pass::CoreFusion::construct_folded_batch_norm()
{
    Shape shape{2, 2, 1, 1};
    auto input = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);

    auto pconv = std::make_shared<op::Convolution>(input,
                                                   filters,
                                                   Strides{1, 1},
                                                   Strides{1, 1},
                                                   CoordinateDiff{0, 0},
                                                   CoordinateDiff{0, 0},
                                                   Strides{1, 1});
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
    auto bn = std::make_shared<op::BatchNormInference>(eps, gamma, beta, pconv, mean, var);

    ngraph::pattern::graph_rewrite_callback callback = [input, filters, mean, var, gamma, beta](
        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for folded batch norm against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto m_bn = std::static_pointer_cast<op::BatchNormInference>(m.get_match_root());
        auto m_conv = std::static_pointer_cast<op::Convolution>(m_bn->get_argument(2));

        if (m_conv->get_users().size() > 1)
        {
            return false;
        }

        if (m_conv->get_shape().size() != 4)
        {
            return false;
        }

        // new weights = old weights * gamma / sqrt(variance + epsilon)
        // new biases = -mean * gamma / sqrt(variance + epsilon) + beta

        auto bn_eps = op::Constant::create(element::f32, Shape{}, {m_bn->get_eps_value()});
        auto var_eps = std::make_shared<op::Add>(
            pattern_map[var],
            std::make_shared<op::Broadcast>(bn_eps, pattern_map[var]->get_shape(), AxisSet{0}));
        auto sqrt_var_eps = std::make_shared<op::Sqrt>(var_eps);

        auto mean_gamma = std::make_shared<op::Multiply>(pattern_map[mean], pattern_map[gamma]);
        auto new_biases = std::make_shared<op::Subtract>(
            pattern_map[beta], std::make_shared<op::Divide>(mean_gamma, sqrt_var_eps));
        auto weight_scaling = std::make_shared<op::Divide>(pattern_map[gamma], sqrt_var_eps);
        auto new_weights = std::make_shared<op::Multiply>(
            pattern_map[filters],
            std::make_shared<op::Broadcast>(
                weight_scaling, pattern_map[filters]->get_shape(), AxisSet{1, 2, 3}));

        auto conv = std::make_shared<op::Convolution>(pattern_map[input],
                                                      new_weights,
                                                      m_conv->get_window_movement_strides(),
                                                      m_conv->get_window_dilation_strides(),
                                                      m_conv->get_padding_below(),
                                                      m_conv->get_padding_above(),
                                                      m_conv->get_data_dilation_strides());
        auto conv_bias =
            conv + std::make_shared<op::Broadcast>(new_biases, conv->get_shape(), AxisSet{0, 2, 3});
        ngraph::replace_node(m.get_match_root(), conv_bias);

        return true;

    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(bn, callback);
    this->add_matcher(m);
}

void pass::CoreFusion::construct_conv_affine_folding()
{
    // A * Conv (input, filters) + B -> ConvBias (input, filters * A_c, B_c)
    Shape shape{2, 2, 1, 1};
    auto input = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);

    auto conv = std::make_shared<op::Convolution>(input,
                                                  filters,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1});
    auto conv_label = std::make_shared<pattern::op::Label>(conv, nullptr, NodeVector{conv});

    auto Ac = std::make_shared<pattern::op::Label>(element::f32, Shape{2});
    auto A = std::make_shared<op::Broadcast>(Ac, Shape{2, 2, 1, 1}, AxisSet{0, 2, 3});
    auto A_label = std::make_shared<pattern::op::Label>(A, nullptr, NodeVector{A});
    auto Bc = std::make_shared<pattern::op::Label>(element::f32, Shape{2});
    auto B = std::make_shared<op::Broadcast>(Bc, Shape{2, 2, 1, 1}, AxisSet{0, 2, 3});
    auto B_label = std::make_shared<pattern::op::Label>(B, nullptr, NodeVector{B});
    auto multiply = std::make_shared<op::Multiply>(conv_label, A_label);
    auto add = std::make_shared<op::Add>(multiply, B_label);

    ngraph::pattern::graph_rewrite_callback callback =
        [input, filters, conv_label, A_label, B_label](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In callback for conv affine folding against node = "
                         << m.get_match_root()->get_name();
            auto pattern_map = m.get_pattern_map();

            auto conv_m = std::static_pointer_cast<op::Convolution>(pattern_map[conv_label]);

            if (conv_m->get_users().size() > 1)
            {
                return false;
            }

            if (conv_m->get_shape().size() != 4)
            {
                return false;
            }

            auto A_m = std::static_pointer_cast<op::Broadcast>(pattern_map[A_label]);
            auto B_m = std::static_pointer_cast<op::Broadcast>(pattern_map[B_label]);

            // Check if values are being broadcast along channel (2nd) dimension
            auto is_channel_bcast = [](const shared_ptr<op::Broadcast>& bcast) {

                if (bcast->get_argument(0)->get_shape().size() == 1 &&
                    bcast->get_broadcast_axes() == AxisSet{0, 2, 3})
                {
                    return true;
                }

                if (bcast->get_argument(0)->get_shape().size() == 2)
                {
                    auto input_shape = bcast->get_argument(0)->get_shape();
                    if (input_shape[0] == 1 && bcast->get_broadcast_axes() == AxisSet{2, 3})
                        return true;
                }
                return false;
            };

            if (!is_channel_bcast(A_m) || !is_channel_bcast(B_m))
            {
                return false;
            }

            auto get_bcast_input = [](const shared_ptr<op::Broadcast>& bcast) {
                if (bcast->get_argument(0)->get_shape().size() == 1)
                {
                    return bcast->get_argument(0);
                }
                if (bcast->get_argument(0)->get_shape().size() == 2)
                {
                    Shape bshape{bcast->get_argument(0)->get_shape()[1]};
                    return static_pointer_cast<ngraph::Node>(std::make_shared<op::Reshape>(
                        bcast->get_argument(0), AxisVector{0, 1}, bshape));
                }
                throw ngraph_error("Unexpected shape for bcast input");
            };

            auto Ac_m = get_bcast_input(A_m);

            // new weights = old weights * Ac_m
            // new biases = Bc_m

            auto filters_n = std::make_shared<op::Multiply>(
                pattern_map[filters],
                std::make_shared<op::Broadcast>(
                    Ac_m, pattern_map[filters]->get_shape(), AxisSet{1, 2, 3}));

            auto conv_n = std::make_shared<op::Convolution>(pattern_map[input],
                                                            filters_n,
                                                            conv_m->get_window_movement_strides(),
                                                            conv_m->get_window_dilation_strides(),
                                                            conv_m->get_padding_below(),
                                                            conv_m->get_padding_above(),
                                                            conv_m->get_data_dilation_strides());
            auto convbias_n = conv_n + B_m;
            ngraph::replace_node(m.get_match_root(), convbias_n);

            return true;

        };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, callback);
    this->add_matcher(m);
}

static bool is_trivial_convolution(std::shared_ptr<op::Convolution> conv,
                                   bool skip_pad_checks = false)
{
    Strides stride_1{1, 1};
    CoordinateDiff pad_0{0, 0};

    return conv->get_window_dilation_strides() == stride_1 &&
           conv->get_data_dilation_strides() == stride_1 &&
           (skip_pad_checks ||
            (conv->get_padding_above() == pad_0 && conv->get_padding_below() == pad_0));
}

static bool are_img_dims_equal(Shape conv_shape, Shape image_shape)
{
    if (conv_shape.size() != 4)
    {
        return false;
    }

    return conv_shape[2] == image_shape[0] && conv_shape[3] == image_shape[1];
}

static std::shared_ptr<Node> reduce_broadcast(std::shared_ptr<Node> broadcast)
{
    const size_t H = 2;
    const size_t W = 3;
    auto matched_broadcast_w1 = std::static_pointer_cast<op::Broadcast>(broadcast);
    Shape shape_w1{matched_broadcast_w1->get_shape()};
    shape_w1[H] /= 2;
    shape_w1[W] /= 2;
    auto new_broadcast_w1 =
        std::make_shared<op::Broadcast>(matched_broadcast_w1->get_argument(0),
                                        shape_w1,
                                        matched_broadcast_w1->get_broadcast_axes());
    return new_broadcast_w1;
}

static size_t shape_to_index(Shape shape)
{
    if (shape.size() != 4)
    {
        return 0;
    }
    const size_t HEIGHT_DIM = 2;
    const size_t WIDTH_DIM = 3;

    if (shape.at(HEIGHT_DIM) != shape.at(WIDTH_DIM))
    {
        return 0;
    }

    switch (shape.at(HEIGHT_DIM))
    {
    case 28: return 1;
    case 14: return 2;
    case 7: return 3;
    default: return 0;
    }
}

//   conv(56w3s1)                        conv(28w3s2)
//	      |                           	    |
//   conv(56w1s1)              ==>      conv(28w1s1)
//       |                                 |
//elt------------56               elt------------pool(28s2)
// |            |                  |               |
//conv(28w1s2) conv(28w1s2)     conv(28w1s1)  conv(28w1s1)
void pass::CoreFusion::construct_optimized_strided_conv()
{
    Shape win_size_1{1, 1, 1, 1};
    auto is_bc = ngraph::pattern::has_class<op::Broadcast>();
    auto data_stride3 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 128, 128});
    auto weights_stride3 = std::make_shared<pattern::op::Label>(element::f32, win_size_1);

    auto conv_stride3 = std::make_shared<op::Convolution>(data_stride3, weights_stride3);

    auto conv_stride3_label =
        std::make_shared<pattern::op::Label>(conv_stride3, nullptr, NodeVector{conv_stride3});

    auto broadcast_w3_label = std::make_shared<pattern::op::Label>(conv_stride3_label, is_bc);
    auto add_w3 = std::make_shared<op::Add>(conv_stride3_label, broadcast_w3_label);
    auto relu_w3 = std::make_shared<op::Relu>(add_w3);

    auto weights_stride1 = std::make_shared<pattern::op::Label>(element::f32, win_size_1);
    auto conv_stride1 = std::make_shared<op::Convolution>(relu_w3, weights_stride1);
    auto conv_stride1_label =
        std::make_shared<pattern::op::Label>(conv_stride1, nullptr, NodeVector{conv_stride1});
    auto broadcast_w1_label = std::make_shared<pattern::op::Label>(conv_stride1_label, is_bc);
    auto add_w1 = std::make_shared<op::Add>(conv_stride1_label, broadcast_w1_label);

    auto eltwise_arg_label =
        std::make_shared<pattern::op::Label>(element::f32, conv_stride1->get_shape());
    auto add_two_convs = std::make_shared<op::Add>(add_w1, eltwise_arg_label);

    auto relu_two_convs = std::make_shared<op::Relu>(add_two_convs);

    auto eltwise_label =
        std::make_shared<pattern::op::Label>(relu_two_convs, nullptr, NodeVector{relu_two_convs});

    auto weights_eltwise = std::make_shared<pattern::op::Label>(element::f32, win_size_1);
    auto eltwise_conv = std::make_shared<op::Convolution>(eltwise_label, weights_eltwise);

    pattern::graph_rewrite_callback callback = [win_size_1,
                                                eltwise_label,
                                                conv_stride1_label,
                                                conv_stride3_label,
                                                eltwise_arg_label,
                                                broadcast_w3_label,
                                                broadcast_w1_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_conv_skip against "
                     << m.get_match_root()->get_name();

        if (m.get_match_root()->get_users().empty())
        {
            NGRAPH_DEBUG << m.get_match_root()
                         << " has already been replaced by a preceding callback";
            return false;
        }

        auto pattern_map = m.get_pattern_map();
        auto m_eltwise = pattern_map[eltwise_label];

        std::vector<std::shared_ptr<Node>> strided_convs;
        for (auto n : m_eltwise->get_users())
        {
            if (is_used(n.get()))
            {
                if (std::dynamic_pointer_cast<op::Convolution>(n) == nullptr)
                {
                    NGRAPH_DEBUG << "Not all live users of element wise operation are Convolution";
                    return false;
                }
                strided_convs.push_back(n);
            }
        }

        if (strided_convs.size() != 2)
        {
            NGRAPH_DEBUG << "Number of live users of element wise operation isn't equal to 2";
            return false;
        }

        Shape supported_shapes[] = {Shape{56, 56}, Shape{28, 28}, Shape{14, 14}, Shape{7, 7}};
        Shape shape_1{1, 1};
        Shape shape_3{3, 3};
        Strides stride_2{2, 2};
        Strides stride_1{1, 1};
        CoordinateDiff pad_0{0, 0};
        CoordinateDiff pad_1{1, 1};
        Shape win_size_3{1, 1, 3, 3};

        size_t sparse_shape_index = 0;
        NodeVector sconvs;
        for (auto sc : strided_convs)
        {
            if (sc->get_argument(0) != m_eltwise)
            {
                NGRAPH_DEBUG << "element-wise isn't data";
                return false;
            }
            auto sconv = std::static_pointer_cast<op::Convolution>(sc);
            sparse_shape_index = shape_to_index(sconv->get_shape());
            if (sparse_shape_index == 0)
            {
                NGRAPH_DEBUG << "Unsupported shape of " << sconv->get_name();
                return false;
            }
            if (!are_img_dims_equal(sconv->get_shape(), supported_shapes[sparse_shape_index]) ||
                !are_img_dims_equal(sconv->get_argument(1)->get_shape(), shape_1) ||
                sconv->get_window_movement_strides() != stride_2 || !is_trivial_convolution(sconv))
            {
                NGRAPH_DEBUG << sconv->get_name() << " and its weights are of the wrong shape (not "
                             << vector_to_string(supported_shapes[sparse_shape_index])
                             << " and 1x1) and strides (2x2)";
                return false;
            }
            sconvs.push_back(sconv);
        }

        const size_t full_shape_index = sparse_shape_index - 1;

        auto m_conv_stride1 =
            std::static_pointer_cast<op::Convolution>(pattern_map[conv_stride1_label]);

        if (!are_img_dims_equal(m_conv_stride1->get_shape(), supported_shapes[full_shape_index]) ||
            !are_img_dims_equal(m_conv_stride1->get_argument(1)->get_shape(), win_size_1) ||
            m_conv_stride1->get_window_movement_strides() != stride_1 ||
            !is_trivial_convolution(m_conv_stride1))
        {
            NGRAPH_DEBUG << m_conv_stride1->get_name()
                         << " and its weights are of the wrong shape (not "
                         << vector_to_string(supported_shapes[full_shape_index])
                         << " and 1x1) and strides (1x1)";
            return false;
        }

        auto m_conv_stride3 =
            std::static_pointer_cast<op::Convolution>(pattern_map[conv_stride3_label]);

        if (!are_img_dims_equal(m_conv_stride3->get_shape(), supported_shapes[full_shape_index]) ||
            !are_img_dims_equal(m_conv_stride3->get_argument(1)->get_shape(), shape_3) ||
            m_conv_stride3->get_window_movement_strides() != stride_1 ||
            !is_trivial_convolution(m_conv_stride3, true))
        {
            NGRAPH_DEBUG << m_conv_stride3->get_name()
                         << " and its weights are of the wrong shape (not "
                         << vector_to_string(supported_shapes[full_shape_index])
                         << " and 3x3) and strides (1x1)";
            return false;
        }

        auto conv_28w3s2 = std::make_shared<op::Convolution>(m_conv_stride3->get_argument(0),
                                                             m_conv_stride3->get_argument(1),
                                                             stride_2,
                                                             stride_1,
                                                             pad_1,
                                                             pad_1);

        auto new_add_conv_28w3s2 = std::make_shared<op::Add>(
            conv_28w3s2, reduce_broadcast(pattern_map[broadcast_w3_label]));
        auto new_relu_28w3s2 = std::make_shared<op::Relu>(new_add_conv_28w3s2);

        auto conv_28w1s1 = std::make_shared<op::Convolution>(
            new_relu_28w3s2, m_conv_stride1->get_argument(1), stride_1, stride_1);

        auto new_add_conv28s1 = std::make_shared<op::Add>(
            conv_28w1s1, reduce_broadcast(pattern_map[broadcast_w1_label]));

        auto maxpool =
            std::make_shared<op::MaxPool>(pattern_map[eltwise_arg_label], Shape{1, 1}, stride_2);
        auto new_add_two_convs = std::make_shared<op::Add>(new_add_conv28s1, maxpool);
        auto new_relu_two_convs = std::make_shared<op::Relu>(new_add_two_convs);

        for (auto sconv : sconvs)
        {
            auto sconv_28w1s1 = std::make_shared<op::Convolution>(
                new_relu_two_convs, sconv->get_argument(1), stride_1, stride_1);
            NGRAPH_DEBUG << "Replacing " << sconv->get_name() << " with "
                         << sconv_28w1s1->get_name();
            ngraph::replace_node(sconv, sconv_28w1s1);
        }
        return true;
    };

    auto m = make_shared<pattern::Matcher>(eltwise_conv, callback);
    this->add_matcher(m);
}
