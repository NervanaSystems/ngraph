/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "constant_propagation.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/reference/reshape.hpp"

using namespace std;
using namespace ngraph;

void ngraph::pass::ConstantPropagation::construct_constant_reshape()
{
    auto constant_pred = [](shared_ptr<Node> n) {
        return dynamic_pointer_cast<op::Constant>(n) != nullptr;
    };

    auto constant_label = make_shared<pattern::op::Label>(element::f32, Shape{2, 4}, constant_pred);
    auto reshape = make_shared<op::Reshape>(constant_label, AxisVector{0, 1}, Shape{2, 4, 1});
    auto reshape_label = make_shared<pattern::op::Label>(reshape, nullptr, NodeVector{reshape});

    auto constant_reshape_callback = [constant_label, reshape_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto constant_match = dynamic_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto reshape_match = dynamic_pointer_cast<op::Reshape>(pattern_map[reshape_label]);

        auto type = constant_match->get_element_type();
        auto in_shape = constant_match->get_shape();
        auto out_shape = reshape_match->get_shape();
        auto order = reshape_match->get_input_order();

        if (type == element::i32)
        {
            auto in_vec = constant_match->get_vector<int>();
            auto out_vec = vector<int>(shape_size(out_shape));
            runtime::reference::reshape<int>(
                in_vec.data(), out_vec.data(), in_shape, order, out_shape);
            auto new_const = make_shared<op::Constant>(type, out_shape, out_vec);
            replace_node(m.get_match_root(), new_const);
            return true;
        }
        else if (type == element::i8)
        {
            auto in_vec = constant_match->get_vector<signed char>();
            auto out_vec = vector<signed char>(shape_size(out_shape));
            runtime::reference::reshape<signed char>(
                in_vec.data(), out_vec.data(), in_shape, order, out_shape);
            auto new_const = make_shared<op::Constant>(type, out_shape, out_vec);
            replace_node(m.get_match_root(), new_const);
            return true;
        }
        else if (type == element::f32)
        {
            auto in_vec = constant_match->get_vector<float>();
            auto out_vec = vector<float>(shape_size(out_shape));
            runtime::reference::reshape<float>(
                in_vec.data(), out_vec.data(), in_shape, order, out_shape);
            auto new_const = make_shared<op::Constant>(type, out_shape, out_vec);
            replace_node(m.get_match_root(), new_const);
            return true;
        }
        else if (type == element::f64)
        {
            auto in_vec = constant_match->get_vector<double>();
            auto out_vec = vector<double>(shape_size(out_shape));
            runtime::reference::reshape<double>(
                in_vec.data(), out_vec.data(), in_shape, order, out_shape);
            auto new_const = make_shared<op::Constant>(type, out_shape, out_vec);
            replace_node(m.get_match_root(), new_const);
            return true;
        }

        return false;
    };

    auto reshape_matcher = make_shared<pattern::Matcher>(reshape_label, constant_reshape_callback);
    this->add_matcher(reshape_matcher);
}
