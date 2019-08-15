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
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/gen/core/add.hpp"
#include "ngraph/op/gen/core/convolution.hpp"

using namespace std;
using namespace ngraph;

TEST(gen_op, description_attributes_add)
{
    auto x = make_shared<op::Parameter>(element::i32, Shape{2, 5});
    auto y = make_shared<op::Parameter>(element::i32, Shape{5});
    auto add = make_shared<op::gen::core::Add>(x, y, op::AutoBroadcastType::NUMPY);
    std::stringstream ss;
    ss << *add;
    EXPECT_PRED_FORMAT2(testing::IsSubstring,
                        "autobroadcast=AutoBroadcastSpec(AutoBroadcastType::NUMPY)",
                        ss.str());
}

TEST(gen_op, description_attributes_convolution)
{
    auto convolution = make_shared<op::gen::core::Convolution>();
    convolution->set_strides(Strides{1, 2, 3});
    convolution->set_dilation(Strides{4, 5, 6});
    convolution->set_data_dilation(Strides{7, 8, 9});
    convolution->set_padding_before(CoordinateDiff{10, 11, 12});
    convolution->set_padding_after(CoordinateDiff{13, 14, 15});
    convolution->set_pad_type(op::PadType::SAME_LOWER);

    std::stringstream ss;
    ss << *convolution;

    EXPECT_PRED_FORMAT2(testing::IsSubstring, "strides=Strides{1, 2, 3}", ss.str());
    EXPECT_PRED_FORMAT2(testing::IsSubstring, "dilation=Strides{4, 5, 6}", ss.str());
    EXPECT_PRED_FORMAT2(testing::IsSubstring, "data_dilation=Strides{7, 8, 9}", ss.str());
    EXPECT_PRED_FORMAT2(
        testing::IsSubstring, "padding_before=CoordinateDiff{10, 11, 12}", ss.str());
    EXPECT_PRED_FORMAT2(testing::IsSubstring, "padding_after=CoordinateDiff{13, 14, 15}", ss.str());
    EXPECT_PRED_FORMAT2(testing::IsSubstring, "pad_type=PadType::SAME_LOWER", ss.str());
}
