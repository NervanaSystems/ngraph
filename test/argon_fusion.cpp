// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
//
#include "ngraph/file_util.hpp"
#include "ngraph/json.hpp"
#include "ngraph/runtime/argon/ops/relu.hpp"
#include "ngraph/runtime/argon/pass/argon_fusion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(Argon_fusion, fuse_max_with_constant_zero_input_as_relu)
{
    auto shape_a = Shape{1, 5};
    auto A = op::Constant::create(element::f32, shape_a, {0, 0, 0, 0, 0});
    auto B = make_shared<op::Parameter>(element::f32, shape_a);
    auto max = make_shared<op::Maximum>(A, B);
    auto shape_rt = Shape{1, 5};
    auto f = make_shared<Function>(max, op::Parameters{B});

    auto manager = runtime::Manager::get("ARGON");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto b = backend->make_primary_tensor_view(element::f32, shape_a);
    copy_data(b, vector<float>{1, 8, -8, 17, -0.5});
    auto result = backend->make_primary_tensor_view(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0};

    cf->call({b}, {result});
    EXPECT_EQ(read_vector<float>(result), expected);
}