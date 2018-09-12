/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <iostream>
#include <numeric>

#include "gtest/gtest.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/activate.hpp"
#include "ngraph/op/deactivate.hpp"
#include "ngraph/op/generate_mask.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/state/rng_state.hpp"
#include "ngraph/util.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(state, stateful)
{
    //Activate(std::shared_ptr<State>& state)
    //Deactivate(const shared_ptr<Node> arg, std::shared_ptr<State>& state)
    //GenerateMask(std::shared_ptr<Node> training, std::shared_ptr<Node> activate, const Shape& shape, const element::Type& element_type, double prob, const std::shared_ptr<RNGState>& state):

    Shape scalar{};
    Shape result_shape{1, 20};
    auto training = op::Constant::create(element::i8, Shape{}, {1});
    auto rng_state = make_shared<RNGState>();
    auto activate = make_shared<op::Activate>(rng_state);
    auto gen_mask = make_shared<op::GenerateMask>(
        training, activate, result_shape, element::i32, 0.5, rng_state);
    auto f = make_shared<Function>(gen_mask, op::ParameterVector{});

    auto backend = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> result_tv = backend->create_tensor<int>(result_shape);
    backend->call_with_validate(f, {result_tv}, {});
    auto result1 = read_vector<int>(result_tv);
    backend->call_with_validate(f, {result_tv}, {});
    auto result2 = read_vector<int>(result_tv);
    ASSERT_EQ(result1, result2);
}
