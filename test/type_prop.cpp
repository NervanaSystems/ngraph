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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, function_revalidate_and_infer)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = op::Constant::create(element::i64, Shape{6}, {1, 3, 16, 2, 2, 2});

    auto r = make_shared<op::DynReshape>(arg, pattern);
    auto relu = make_shared<op::Relu>(r);
    auto f = make_shared<Function>(relu, ParameterVector{arg});

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_shape(0), (Shape{1, 3, 16, 2, 2, 2}));
    EXPECT_EQ(f->get_output_shape(0), (Shape{1, 3, 16, 2, 2, 2}));

    auto new_pattern = op::Constant::create(element::i64, Shape{2}, {32, 12});
    r->input(1).replace_source_output(new_pattern->output(0));

    f->validate_nodes_and_infer_types();
    EXPECT_EQ(r->get_output_shape(0), (Shape{32, 12}));
    EXPECT_EQ(f->get_output_shape(0), (Shape{32, 12}));
}