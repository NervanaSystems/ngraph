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

#include "ngraph/runtime/plaidml/plaidml_pass_winograd.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/plaidml/plaidml_ops_convolution.hpp"
#include "ngraph/runtime/plaidml/plaidml_ops_winograd.hpp"

namespace
{
    std::tuple<std::shared_ptr<ngraph::op::Constant>,
               std::shared_ptr<ngraph::op::Constant>,
               std::shared_ptr<ngraph::op::Constant>>
        make_l_4_3()
    {
        std::vector<float> a_vec{1.13777777777778,
                                 0,
                                 0,
                                 0,
                                 -0.688403361344538,
                                 -0.430252100840336,
                                 -0.26890756302521,
                                 -0.168067226890756,
                                 -0.688403361344538,
                                 0.430252100840336,
                                 -0.26890756302521,
                                 0.168067226890756,
                                 0.119514472455649,
                                 0.179271708683473,
                                 0.26890756302521,
                                 0.403361344537815,
                                 0.119514472455649,
                                 -0.179271708683473,
                                 0.26890756302521,
                                 -0.403361344537815,
                                 0,
                                 0,
                                 0,
                                 1};
        auto a = std::make_shared<ngraph::op::Constant>(
            ngraph::element::f32, ngraph::Shape{6, 4}, std::move(a_vec));

        std::vector<float> b_vec{0.87890625, 0,        0,       0,          0,         0,
                                 0,          -1.40625, 1.40625, -0.5859375, 0.5859375, 0.87890625,
                                 -2.640625,  -2.25,    -2.25,   -0.390625,  -0.390625, 0,
                                 0,          0.625,    -0.625,  1.5,        -1.5,      -2.640625,
                                 1,          1,        1,       1,          1,         0,
                                 0,          0,        0,       0,          0,         1};
        auto b = std::make_shared<ngraph::op::Constant>(
            ngraph::element::f32, ngraph::Shape{6, 6}, std::move(b_vec));

        std::vector<float> g_vec{
            1, 0, 0, 1, 0.625, 0.390625, 1, -0.625, 0.390625, 1, 1.5, 2.25, 1, -1.5, 2.25, 0, 0, 1};
        auto g = std::make_shared<ngraph::op::Constant>(
            ngraph::element::f32, ngraph::Shape{6, 3}, std::move(g_vec));

        return std::make_tuple(std::move(a), std::move(b), std::move(g));
    }

    std::tuple<std::shared_ptr<ngraph::op::Constant>,
               std::shared_ptr<ngraph::op::Constant>,
               std::shared_ptr<ngraph::op::Constant>>
        l_4_3()
    {
        static std::tuple<std::shared_ptr<ngraph::op::Constant>,
                          std::shared_ptr<ngraph::op::Constant>,
                          std::shared_ptr<ngraph::op::Constant>>
            params = make_l_4_3();

        return params;
    }
}

ngraph::runtime::plaidml::pass::Winograd::Winograd()
{
    auto convolution_op =
        std::make_shared<pattern::op::Label>(element::i8, Shape{}, [](std::shared_ptr<Node> node) {
            auto* conv = dynamic_cast<plaidml::op::Convolution*>(node.get());
            if (!conv)
            {
                return false;
            }
            // Our Winograd implementation is limited to specific
            // cases where we think it results in a performance
            // improvement, and cases that were straightforward to
            // implement -- hence all the conditions.  It may be
            // useful to extend the implementation to cover additional
            // cases.
            const auto& data_shape = conv->get_input_shape(0);
            const auto& filters_shape = conv->get_input_shape(1);
            return (data_shape.size() == 4 && conv->get_data_axes() == AxisVector{0, 3, 1, 2} &&
                    conv->get_filters_axes() == AxisVector{3, 2, 0, 1} &&
                    conv->get_output_axes() == AxisVector{0, 2, 3, 1} &&
                    conv->get_src()->get_window_movement_strides() == Strides{1, 1} &&
                    conv->get_src()->get_window_dilation_strides() == Strides{1, 1} &&
                    filters_shape.size() >= 4 && filters_shape.at(0) == 3 &&
                    filters_shape.at(1) == 3 && filters_shape.at(2) > 4 && filters_shape.at(3) > 4);
        });

    auto callback = [](pattern::Matcher& m) {
        auto conv = std::static_pointer_cast<plaidml::op::Convolution>(m.get_match_root());
        OutputVector args;
        for (auto input : conv->inputs())
        {
            args.push_back(input.get_source_output());
        }
        std::shared_ptr<ngraph::op::Constant> a;
        std::shared_ptr<ngraph::op::Constant> b;
        std::shared_ptr<ngraph::op::Constant> g;
        std::tie(a, b, g) = l_4_3(); // N.B. => filters HW must be 3x3
        args.emplace_back(a);
        args.emplace_back(b);
        args.emplace_back(g);
        auto winograd = std::make_shared<plaidml::op::Winograd>(conv, args);
        replace_node(std::move(conv), std::move(winograd));
        return true;
    };

    add_matcher(std::make_shared<pattern::Matcher>(convolution_op), callback);
}
