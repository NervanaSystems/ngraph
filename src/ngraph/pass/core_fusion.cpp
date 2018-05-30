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
