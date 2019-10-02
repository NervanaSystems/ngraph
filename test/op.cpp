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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/variant.hpp"

using namespace std;
using namespace ngraph;

TEST(op, is_op)
{
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    EXPECT_TRUE(arg0->is_parameter());
}

TEST(op, is_parameter)
{
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
    ASSERT_NE(nullptr, arg0);
    auto t0 = make_shared<op::Add>(arg0, arg0);
    ASSERT_NE(nullptr, t0);
    EXPECT_FALSE(t0->is_parameter());
}

TEST(op, provenance_tag)
{
    auto node = make_shared<op::Parameter>(element::f32, Shape{1});
    auto tag1 = "parameter node";
    auto tag2 = "f32 node";
    node->add_provenance_tag(tag1);
    node->add_provenance_tag(tag2);

    node->remove_provenance_tag(tag1);

    auto tags = node->get_provenance_tags();
    ASSERT_TRUE(tags.find(tag1) == tags.end());
    ASSERT_TRUE(tags.find(tag2) != tags.end());
}

struct Ship
{
    std::string name;
    int16_t x;
    int16_t y;
};

DECLARE_VARIANT(Ship, , "Variant::Ship", 0);
DEFINE_VARIANT(Ship, 0);

TEST(op, variant)
{
    shared_ptr<Variant> var_std_string = make_shared<VARIANT_NAME(std::string, 0)>("My string");
    ASSERT_TRUE((is_type<VARIANT_NAME(std::string, 0)>(var_std_string)));
    EXPECT_EQ((as_type_ptr<VARIANT_NAME(std::string, 0)>(var_std_string)->get()), "My string");

    shared_ptr<Variant> var_uint64_t = make_shared<VARIANT_NAME(uint64_t, 0)>(27);
    ASSERT_TRUE((is_type<VARIANT_NAME(uint64_t, 0)>(var_uint64_t)));
    EXPECT_FALSE((is_type<VARIANT_NAME(std::string, 0)>(var_uint64_t)));
    EXPECT_EQ((as_type_ptr<VARIANT_NAME(uint64_t, 0)>(var_uint64_t)->get()), 27);

    shared_ptr<Variant> var_string = make_shared<StringVariant>("My other string");
    ASSERT_TRUE((is_type<StringVariant>(var_string)));
    EXPECT_EQ((as_type_ptr<StringVariant>(var_string)->get()), "My other string");

    shared_ptr<Variant> var_ship = make_shared<VARIANT_NAME(Ship, 0)>(Ship{"Lollipop", 3, 4});
    ASSERT_TRUE((is_type<VARIANT_NAME(Ship, 0)>(var_ship)));
    Ship& ship = as_type_ptr<VARIANT_NAME(Ship, 0)>(var_ship)->get();
    EXPECT_EQ(ship.name, "Lollipop");
    EXPECT_EQ(ship.x, 3);
    EXPECT_EQ(ship.y, 4);

    auto node = make_shared<op::Parameter>(element::f32, Shape{1});
    node->set_rt_info(var_ship);
    auto node_var_ship = node->get_rt_info();
    ASSERT_TRUE((is_type<VARIANT_NAME(Ship, 0)>(node_var_ship)));
    Ship& node_ship = as_type_ptr<VARIANT_NAME(Ship, 0)>(node_var_ship)->get();
    EXPECT_EQ(&node_ship, &ship);
}

// TODO: Need to mock Node, Op etc to be able to unit test functions like replace_node().
// Mocking them directly isn't possible because google test requires methods to be
// non-virtual. For non-virtual methods we will need to templatize these classes and call using
// different template argument between testing and production.
/*
TEST(op, provenance_replace_node)
{
    class MockOp: public op::Op
    {
        MOCK_CONST_METHOD1(copy_with_new_args, std::shared_ptr<Node>(const NodeVector& new_args));
        MOCK_CONST_METHOD1(get_users, NodeVector (bool check_is_used)); // This can't be mocked as
                                                                        // it's non-virtual
    };

    ::testing::NiceMock<MockOp> mock_op;
}
*/
