// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

#include "ngraph/runtime/eigen/add.hpp"
#include "ngraph/runtime/eigen/multiply.hpp"
#include "ngraph/runtime/eigen/return.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::runtime;
namespace ngeigen = ngraph::runtime::eigen;

TEST(runtime, test_add)
{
    auto x = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});
    x->get_vector() = {1, 2, 3, 4};
    auto y = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});
    y->get_vector() = {5, 6, 7, 8};
    auto z = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});
    ngraph::runtime::eigen::add(x, y, z);
    ASSERT_EQ((vector<float>{6, 8, 10, 12}), z->get_vector());
}

TEST(runtime, test_multiply)
{
    auto x = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});
    x->get_vector() = {1, 2, 3, 4};
    auto y = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});
    y->get_vector() = {5, 6, 7, 8};
    auto z = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});
    ngraph::runtime::eigen::multiply(x, y, z);
    ASSERT_EQ((vector<float>{5, 12, 21, 32}), z->get_vector());
}

TEST(runtime, test_add_multiply)
{
    // Inputs:
    //   0 : a
    //   1 : b
    //   2 : c
    // Outputs:
    //   3 : result
    // Temporaries
    //   4: t0
    auto instructions = make_shared<std::vector<std::shared_ptr<ngraph::runtime::Instruction>>>();
    // a + b -> t0
    instructions->push_back(make_shared<ngeigen::AddInstruction<element::Float32>>(0, 1, 4));
    // t0 * c -> result
    instructions->push_back(make_shared<ngeigen::MultiplyInstruction<element::Float32>>(4, 2, 3));
    instructions->push_back(make_shared<ngeigen::ReturnInstruction>());

    runtime::CallFrame cf{
        3, 1, {ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2})}, 0, instructions};

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});
    a->get_vector() = {1, 2, 3, 4};
    auto b = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});
    b->get_vector() = {5, 6, 7, 8};
    auto c = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});
    c->get_vector() = {9, 10, 11, 12};
    auto result = ngraph::runtime::make_tensor<element::Float32>(Shape{2, 2});

    cf({a, b, c}, {result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    cf({b, a, c}, {result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    cf({a, c, b}, {result});
    ASSERT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector());
}
