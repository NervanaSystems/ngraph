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

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/ndarray.hpp"

using namespace std;
using namespace ngraph;

TEST(replace_node, replace_by_friendly_name)
{
    auto x = make_shared<op::Parameter>(element::f32, Shape{2});
    x->set_friendly_name("x");
    auto y = make_shared<op::Parameter>(element::f32, Shape{2});
    y->set_friendly_name("y");
    auto z = make_shared<op::Parameter>(element::f32, Shape{2});
    z->set_friendly_name("z");

    auto add = x + y;
    add->set_friendly_name("add");
    auto k = make_shared<op::Constant>(element::f32, Shape{2}, vector<float>{1, 2});
    k->set_friendly_name("k");
    auto mul = add * k;
    mul->set_friendly_name("mul");
    auto sub = mul - z;
    sub->set_friendly_name("sub");

    auto f = make_shared<Function>(NodeVector{sub}, ParameterVector{x, y, z});

    unordered_map<string, shared_ptr<Node>> replacement_map;

    auto y_replacement = make_shared<op::Constant>(element::f32, Shape{2}, vector<float>{3, 4});
    replacement_map["y"] = y_replacement;

    auto k_replacement = make_shared<op::Constant>(element::f32, Shape{2}, vector<float>{5, 6});
    replacement_map["k"] = k_replacement;

    auto z_replacement = x + mul;
    replacement_map["z"] = z_replacement;

    size_t n_replaced = replace_by_friendly_name(f, replacement_map);

    // Should have replaced three nodes.
    ASSERT_EQ(n_replaced, 3);

    // Should still have three params.
    ASSERT_EQ(f->get_parameters().size(), 3);

    // The three params should still be {x, y, z}.
    ASSERT_EQ(f->get_parameters()[0], x);
    ASSERT_EQ(f->get_parameters()[1], y);
    ASSERT_EQ(f->get_parameters()[2], z);

    // y, z should be dead.
    ASSERT_EQ(y->get_users(true).size(), 0);
    ASSERT_EQ(z->get_users(true).size(), 0);

    // Should still have one result.
    ASSERT_EQ(f->get_results().size(), 1);

    // Result node should be sub (unchanged).
    ASSERT_EQ(f->get_results()[0]->input(0).get_source_output().get_node_shared_ptr(), sub);

    // sub's arguments should be mul (unchanged) and z_replacement.
    ASSERT_EQ(sub->input(0).get_source_output().get_node_shared_ptr(), mul);
    ASSERT_EQ(sub->input(1).get_source_output().get_node_shared_ptr(), z_replacement);

    // mul's arguments should be add (unchanged) and k_replacement.
    ASSERT_EQ(mul->input(0).get_source_output().get_node_shared_ptr(), add);
    ASSERT_EQ(mul->input(1).get_source_output().get_node_shared_ptr(), k_replacement);

    // add's arguments should be x (unchanged) and y_replacement.
    ASSERT_EQ(add->input(0).get_source_output().get_node_shared_ptr(), x);
    ASSERT_EQ(add->input(1).get_source_output().get_node_shared_ptr(), y_replacement);

    // z_replacement's arguments should both be x.
    ASSERT_EQ(z_replacement->input(0).get_source_output().get_node_shared_ptr(), x);
    ASSERT_EQ(z_replacement->input(0).get_source_output().get_node_shared_ptr(), x);

    // Replacements should have inherited friendly names from their replacees.
    ASSERT_EQ(y_replacement->get_friendly_name(), "y");
    ASSERT_EQ(k_replacement->get_friendly_name(), "k");
    ASSERT_EQ(z_replacement->get_friendly_name(), "z");

    // Replacees should have friendly names blanked out, which results in them reverting to their
    // "unfriendly" names.
    ASSERT_EQ(y->get_friendly_name(), y->get_name());
    ASSERT_EQ(k->get_friendly_name(), k->get_name());
    ASSERT_EQ(z->get_friendly_name(), z->get_name());
}
