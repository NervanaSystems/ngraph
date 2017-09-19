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

#include "ngraph.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::runtime;
namespace ngeigen = ngraph::runtime::eigen;

TEST(runtime, test_add)
{
    auto x = make_shared<ngeigen::PrimaryTensorView<element::Float32>>(Shape{2, 2});
    *x     = std::vector<float>{1, 2, 3, 4};
    auto y = make_shared<ngeigen::PrimaryTensorView<element::Float32>>(Shape{2, 2});
    *y     = std::vector<float>{5, 6, 7, 8};
    auto z = make_shared<ngeigen::PrimaryTensorView<element::Float32>>(Shape{2, 2});
    add(*x, *y, *z);
    ASSERT_EQ((vector<float>{6, 8, 10, 12}), z->get_vector());
}

TEST(runtime, test_multiply)
{
    auto x          = make_shared<op::Float32TensorConstant>(Shape{2, 2});
    *x->get_value() = std::vector<float>{1, 2, 3, 4};
    auto y          = make_shared<op::Float32TensorConstant>(Shape{2, 2});
    *y->get_value() = std::vector<float>{5, 6, 7, 8};
    auto z          = make_shared<op::Float32TensorConstant>(Shape{2, 2});
    multiply(*x->get_value(), *y->get_value(), *z->get_value());
    ASSERT_EQ((vector<float>{5, 12, 21, 32}), z->get_value()->get_vector());
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
        3,
        1,
        PTVs{make_shared<ngeigen::PrimaryTensorView<element::Float32>>(Shape{2, 2})},
        0,
        instructions};

    // Create some tensors for input/output
    auto a      = make_shared<ngeigen::PrimaryTensorView<element::Float32>>(Shape{2, 2});
    *a          = vector<float>{1, 2, 3, 4};
    auto b      = make_shared<ngeigen::PrimaryTensorView<element::Float32>>(Shape{2, 2});
    *b          = vector<float>{5, 6, 7, 8};
    auto c      = make_shared<ngeigen::PrimaryTensorView<element::Float32>>(Shape{2, 2});
    *c          = vector<float>{9, 10, 11, 12};
    auto result = make_shared<ngeigen::PrimaryTensorView<element::Float32>>(Shape{2, 2});

    cf(PTVs{a, b, c}, PTVs{result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    cf(PTVs{b, a, c}, PTVs{result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    cf(PTVs{a, c, b}, PTVs{result});
    ASSERT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector());
}
