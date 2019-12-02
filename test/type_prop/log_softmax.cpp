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
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, log_softmax)
{
    const auto data = make_shared<op::Parameter>(element::f64, Shape{2, 2});
    const auto axis = 2;
    try
    {
        const auto log_softmax = make_shared<op::LogSoftmax>(data, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid axis value not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis "));
    }
    catch (...)
    {
        FAIL() << "Log softmax failed for unexpected reason";
    }
}
