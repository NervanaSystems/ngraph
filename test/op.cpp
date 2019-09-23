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
