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

// Standalone Goole Test example for ngraph.
// compile and test as follows.
// g++ -std=c++11 simple_gtest.cpp -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -pthread -lngraph -lgtest -o /tmp/test
// env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib /tmp/test

#include <gtest/gtest.h>
#include <ngraph/ngraph.hpp>
#include "../../test/util/test_tools.hpp"
#include "nutils.hpp"
using namespace std;
using namespace ngraph;

TEST(simple, mul_forward)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto o = ngraph::builder::make_with_numpy_broadcast<op::Multiply>(A, B);
    auto f = make_shared<Function>(o, op::Parameters{A, B});

    CFrame cf;
    TViews inp, out;
    tie(cf, ignore, inp, out) = get_cfio("CPU", f);

    copy_data(inp[0], vector<float>{1, 2, 3, 4});
    copy_data(inp[1], vector<float>{1, 2, 3, 4});
    cf->call(inp, out);

    EXPECT_EQ((vector<float>{1, 4, 9, 16}), read_vector<float>(out[0]));
}

TEST(simple, div_forward_backward)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto o = ngraph::builder::make_with_numpy_broadcast<op::Divide>(A, B);
    auto f = make_shared<Function>(o, op::Parameters{A, B});

    CFrame cf, cb;
    TViews inp, out;
    tie(cf, cb, inp, out) = get_cfio("INTERPRETER", f, true);

    copy_data(inp[0], vector<float>{2, 4, 8, 16});
    copy_data(inp[1], vector<float>{1, 2, 4, 8});
    cf->call(inp, {out[0]});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), read_vector<float>(out[0]));

    copy_data(inp[2], vector<float>{1, 1, 1, 1});
    cb->call({inp[2], inp[0], inp[1]}, {out[1], out[2]});
    EXPECT_EQ((vector<float>{1, 0.5, 0.25, 0.125}), read_vector<float>(out[1]));
    EXPECT_EQ((vector<float>{-2, -1, -0.5, -0.25}), read_vector<float>(out[2]));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
