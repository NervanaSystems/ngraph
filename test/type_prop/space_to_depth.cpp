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

TEST(type_prop, space_to_depth)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    const auto mode = ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, mode, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 128, 8, 8}));
}

TEST(type_prop, space_to_depth_input_rank_not_supported)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 8, 8, 8, 4});
    try
    {
        auto space_to_depth =
            make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
        FAIL() << "Not supported input shape for SpaceToDepth exception not thrown";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "The provided tensor shape: ");
    }
    catch (...)
    {
        FAIL() << "SpaceToDepth decomposition failed for unexpected reason";
    }
}
