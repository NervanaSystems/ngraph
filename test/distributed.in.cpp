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

#include <fstream>
#include <sstream>

#include "gtest/gtest.h"

#include "distributed_setup.hpp"
#include "ngraph/distributed.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "util/random.hpp"

using namespace std;
using namespace ngraph;

TEST(distributed_${BACKEND_NAME}, allreduce)
{
    DistributedSetup distsetup;
    auto comm_size = distsetup.get_comm_size();
    if (comm_size > 1)
    {
        auto shape = Shape{2, 2};
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::AllReduce>(A), ParameterVector{A});

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        auto v = vector<float>{1, 2, 3, 4};
        auto a = backend->create_tensor(element::f32, shape);
        copy_data(a, vector<float>{1, 2, 3, 4});

        auto result = backend->create_tensor(element::f32, shape);

        std::transform(
            v.begin(), v.end(), v.begin(), std::bind1st(std::multiplies<float>(), comm_size));

        auto handle = backend->compile(f);
        handle->call_with_validate({result}, {a});
        EXPECT_EQ(v, read_vector<float>(result));
    }
}
