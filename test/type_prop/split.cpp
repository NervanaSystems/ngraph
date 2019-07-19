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

TEST(type_prop, split)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});

    try
    {
        const std::vector<size_t> splits = {1, 6}; // should sum up to 6
        const auto split = make_shared<op::Split>(data, 1, splits);
        FAIL() << "Split node was created with incorrect data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("has to be equal to the sum of splits passed to the op: 7"));
    }

    try
    {
        const std::vector<size_t> splits = {4, 2};
        const auto split = make_shared<op::Split>(data, -5, splits); //invalid axis
        FAIL() << "Split node was created with incorrect data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The 'axis' parameter for Split has to point to one of "
                                         "the input tensor's shape dimensions."));
    }

    const auto split = make_shared<op::Split>(data, 1, 2);
    EXPECT_EQ(split->outputs().size(), 2);
    EXPECT_EQ(split->output(0).get_shape(), (Shape{2, 3}));
    EXPECT_EQ(split->output(1).get_shape(), (Shape{2, 3}));
    EXPECT_EQ(split->output(0).get_element_type(), element::i32);
    EXPECT_EQ(split->output(1).get_element_type(), element::i32);
}
