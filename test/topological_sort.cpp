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
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/topological_sort.hpp"
#include "ngraph/visualize.hpp"

using namespace std;
using namespace ngraph;

TEST(top_sort, basic)
{
    auto arg0 = op::parameter(element::Float::element_type(), {1});
    ASSERT_NE(nullptr, arg0);
    auto arg1 = op::parameter(element::Float::element_type(), {1});
    ASSERT_NE(nullptr, arg1);
    auto t0 = op::add(arg0, arg1);
    ASSERT_NE(nullptr, t0);
    auto t1 = op::add(arg0, arg1);
    ASSERT_NE(nullptr, t1);
    auto r0 = op::add(t0, t1);
    ASSERT_NE(nullptr, r0);

    auto f0 = op::function(r0, {arg0, arg1});
    ASSERT_NE(nullptr, f0);

    ASSERT_EQ(2, r0->get_arguments().size());
    auto op_r0 = static_pointer_cast<Op>(r0);
    cout << "op_r0 name " << *r0 << endl;

    Visualize vz;
    vz.add(r0);
    vz.save_dot("test.png");
    TopologicalSort::process(r0);
}
