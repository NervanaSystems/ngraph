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

using namespace std;
using namespace ngraph;
using namespace ngraph::runtime::eigen;

TEST(runtime, test_add)
{
    auto x = make_shared<PrimaryTensorView<element::Float32>>(Shape{2, 2});
    *x     = std::vector<float>{1, 2, 3, 4};
    auto y = make_shared<PrimaryTensorView<element::Float32>>(Shape{2, 2});
    *y     = std::vector<float>{5, 6, 7, 8};
    auto z = make_shared<PrimaryTensorView<element::Float32>>(Shape{2, 2});
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
