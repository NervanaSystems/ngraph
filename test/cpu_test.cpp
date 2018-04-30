/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"
#include "ngraph/op/reverse_sequence.hpp"

using namespace ngraph;
using namespace std;

class UnhandledOp : public ngraph::op::Abs
{
public:
    UnhandledOp(const std::shared_ptr<Node>& arg)
        : Abs(arg)
    {
    }
};

TEST(cpu_test, unhandled_op)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{});
    auto unhandled = make_shared<UnhandledOp>(A);
    auto f = make_shared<Function>(unhandled, op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    ASSERT_THROW(backend->compile(f), ngraph_error);
}

TEST(cpu_test, reverse_sequence)
{
    Shape shape{2, 3, 4, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);

    size_t batch_axis = 2;
    size_t sequence_axis = 1;
    Shape sequence_lengths{1,2,1,2};
    auto rs = std::make_shared<op::ReverseSequence>(A, batch_axis, sequence_axis, sequence_lengths);

    auto f = make_shared<Function>(rs, op::ParameterVector{A});

    auto backend = runtime::Backend::create("CPU");

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::i32, shape);
    shared_ptr<runtime::TensorView> result = backend->create_tensor(element::i32, shape);

    std::vector<int> input
    {
                        //B1
                        0,0, 3,0, 6,0, 9,0,
                        1,0, 4,0, 7,0, 10,0,
                        2,0, 5,0, 8,0, 11,0,
                        //B2
                        12,0, 15,0, 18,0, 21,0,
                        13,0, 16,0, 19,0, 22,0,
                        14,0, 17,0, 20,0, 23,0,
    };

    std::vector<int> expected
    {
        0,  0,  4,  0,  6,  0, 10,  0,
        1,  0,  3,  0,  7,  0,  9,  0,
        2,  0,  5,  0,  8,  0, 11,  0,

        12,  0, 16,  0, 18,  0, 22,  0,
        13,  0, 15,  0, 19,  0, 21,  0,
        14,  0, 17,  0, 20,  0, 23,  0
    };

    copy_data(a, input);

    backend->call(f, {result}, {a});
    EXPECT_EQ(read_vector<int>(result), expected);
}
