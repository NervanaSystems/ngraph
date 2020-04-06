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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, non_zero)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 3, 224, 224});
    auto non_zero = make_shared<op::NonZero>(data);
    EXPECT_EQ(non_zero->get_element_type(), element::i64);
    EXPECT_TRUE(
        non_zero->get_output_partial_shape(0).same_scheme(PartialShape{4, Dimension::dynamic()}));
}

TEST(type_prop, non_zero_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto non_zero = make_shared<op::NonZero>(data);
    EXPECT_EQ(non_zero->get_element_type(), element::i64);
    EXPECT_TRUE(non_zero->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
}
