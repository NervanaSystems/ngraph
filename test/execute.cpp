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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(execute, test_abc)
{
    auto shape = Shape{2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Multiply>(make_shared<op::Add>(A, B), C),
                                   op::Parameters{A, B, C});

    auto external = make_shared<ngraph::runtime::eigen::ExternalFunction>(f);
    auto cf = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = make_shared<runtime::eigen::PrimaryTensorView<element::Float32>>(shape);
    *a          = vector<float>{1, 2, 3, 4};
    auto b      = make_shared<runtime::eigen::PrimaryTensorView<element::Float32>>(shape);
    *b          = vector<float>{5, 6, 7, 8};
    auto c      = make_shared<runtime::eigen::PrimaryTensorView<element::Float32>>(shape);
    *c          = vector<float>{9, 10, 11, 12};
    auto result = make_shared<runtime::eigen::PrimaryTensorView<element::Float32>>(shape);

    (*cf)(runtime::PTVs{a, b, c}, runtime::PTVs{result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)(runtime::PTVs{b, a, c}, runtime::PTVs{result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)(runtime::PTVs{a, c, b}, runtime::PTVs{result});
    ASSERT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector());
}
