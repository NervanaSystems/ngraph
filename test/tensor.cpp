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

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::descriptor;

TEST(tensor, size)
{
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::TopologicalSort>();
    pass_manager.register_pass<pass::Liveness>();

    {
        auto arg0 = make_shared<op::Parameter>(element::f32, Shape{2, 3});
        auto add = make_shared<op::Add>(arg0, arg0);
        auto rt = make_shared<TensorViewType>(element::f32, Shape{2, 3});
        auto f0 = make_shared<Function>(add, rt, op::Parameters{arg0});

        pass_manager.run_passes(f0);

        auto& outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(2 * 3 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::f32, Shape{});
        auto add = make_shared<op::Add>(arg0, arg0);
        auto rt = make_shared<TensorViewType>(element::f32, Shape{});
        auto f0 = make_shared<Function>(add, rt, op::Parameters{arg0});

        pass_manager.run_passes(f0);

        auto& outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(1 * 4, output.size());
    }

    {
        auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
        auto add = make_shared<op::Add>(arg0, arg0);
        auto rt = make_shared<TensorViewType>(element::f32, Shape{1});
        auto f0 = make_shared<Function>(add, rt, op::Parameters{arg0});

        pass_manager.run_passes(f0);

        auto& outputs = arg0->get_outputs();
        ASSERT_EQ(1, outputs.size());
        Tensor& output = outputs[0].get_tensor();
        EXPECT_EQ(1 * 4, output.size());
    }
}

template <typename ET>
void test_read_write(const std::vector<typename ET::type>& x)
{
    using T = typename ET::type;

    auto manager = ngraph::runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    auto a = backend->make_primary_tensor_view(ET::element_type(), Shape{2, x.size()});

    std::vector<T> result(2 * x.size());

    a->write(&x[0], 0, x.size() * sizeof(T));
    std::copy(x.begin(), x.end(), result.begin());
    a->write(&x[0], x.size() * sizeof(T), x.size() * sizeof(T));
    std::copy(x.begin(), x.end(), result.begin() + x.size());

    std::vector<T> af_vector(2 * x.size());
    a->read(af_vector.data(), 0, af_vector.size() * sizeof(typename ET::type));
    ASSERT_EQ(af_vector, result);

    std::vector<T> result1(x.size());
    std::vector<T> result2(x.size());
    std::copy(result.begin() + 1, result.begin() + 1 + x.size(), result1.begin());
    a->read(&result2[0], sizeof(T), sizeof(T) * x.size());
    ASSERT_EQ(result1, result2);
}

TEST(tensor, read_write)
{
    test_read_write<element::Float32>({1.0, 3.0, 5.0});
    test_read_write<element::Int64>({-1, 2, 4});
}

TEST(tensor, output_flag)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::TopologicalSort>();
    pass_manager.register_pass<pass::Liveness>();

    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{1});
    auto add = make_shared<op::Add>(arg0, arg0);
    auto rt = make_shared<TensorViewType>(element::f32, Shape{1});
    auto f0 = make_shared<Function>(add, rt, op::Parameters{arg0});

    pass_manager.run_passes(f0);

    EXPECT_TRUE(f0->get_result()->is_output());
    for (descriptor::Output& output : f0->get_result()->get_outputs())
    {
        const Tensor& t = output.get_tensor();
        EXPECT_TRUE(t.is_output());
    }
}
