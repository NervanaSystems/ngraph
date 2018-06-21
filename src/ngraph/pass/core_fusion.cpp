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
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
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

    auto broadcast_pred = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::Broadcast>(n));
    };
    auto skip_broadcast = std::make_shared<pattern::op::Skip>(zero, broadcast_pred);
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
    auto bn = std::make_shared<op::BatchNorm>(eps, gamma, beta, pconv, mean, var);

    ngraph::pattern::graph_rewrite_callback callback = [input, filters, mean, var, gamma, beta](
        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for folded batch norm against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto m_bn = std::dynamic_pointer_cast<op::BatchNorm>(m.get_match_root());
        auto m_conv = std::dynamic_pointer_cast<op::Convolution>(m_bn->get_argument(2));

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
        auto strided_convs = m_eltwise->get_users();
        if (strided_convs.size() != 2)
        {
            NGRAPH_DEBUG << "Number of users of element wise operation isn't equal to 2";
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
            auto sconv = std::dynamic_pointer_cast<op::Convolution>(sc);
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
            std::dynamic_pointer_cast<op::Convolution>(pattern_map[conv_stride1_label]);

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
            std::dynamic_pointer_cast<op::Convolution>(pattern_map[conv_stride3_label]);

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

        auto maxpool_w3 =
            std::make_shared<op::MaxPool>(pattern_map[broadcast_w3_label], Shape{1, 1}, stride_2);
        auto new_add_conv_28w3s2 = std::make_shared<op::Add>(conv_28w3s2, maxpool_w3);
        auto new_relu_28w3s2 = std::make_shared<op::Relu>(new_add_conv_28w3s2);

        auto conv_28w1s1 = std::make_shared<op::Convolution>(
            new_relu_28w3s2, m_conv_stride1->get_argument(1), stride_1, stride_1);

        auto maxpool_w1 =
            std::make_shared<op::MaxPool>(pattern_map[broadcast_w1_label], Shape{1, 1}, stride_2);
        auto new_add_conv28s1 = std::make_shared<op::Add>(conv_28w1s1, maxpool_w1);

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
