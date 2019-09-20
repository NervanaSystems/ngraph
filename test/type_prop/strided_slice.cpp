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

TEST(type_prop, strided_slice_begin_incorrect_type)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i32, Shape{4});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    try
    {
        auto strided_slice = make_shared<op::StridedSlice>(data, begin, end, AxisSet{1, 0, 1, 0}, AxisSet{ 1, 0, 1, 0 });
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect begin type exception not thrown.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Begin mask must have element type i64."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_end_incorrect_type)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto end = make_shared<op::Parameter>(element::i32, Shape{4});
    try
    {
        auto strided_slice = make_shared<op::StridedSlice>(data, begin, end, AxisSet{ 1, 0, 1, 0 }, AxisSet{ 1, 0, 1, 0 });
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect end type exception not thrown.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("End mask must have element type i64."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, strided_slice_incompatible_size_of_masks_attr)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{ 2, 4, 6, 8 });
    auto begin = make_shared<op::Parameter>(element::i64, Shape{ 4 });
    auto end = make_shared<op::Parameter>(element::i64, Shape{ 4 });
    try
    {
        auto strided_slice = make_shared<op::StridedSlice>(data, begin, end,
            AxisSet{ 1, 0, 1, 0 },
            AxisSet{ 1, 0, 1, 0 },
            AxisSet{ 1, 0, 1, 0, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible size od masks exception not thrown.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("All maks of StridedSlice should have the same size"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
