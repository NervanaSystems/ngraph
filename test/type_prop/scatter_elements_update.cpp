//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

TEST(type_prop, scatter_elements_update_check_output_shape)
{
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2, 2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};
    Shape expected_output_shape{2, 4, 5, 7};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);

    EXPECT_EQ(scatter->get_output_shape(0), expected_output_shape);
}

TEST(type_prop, scatter_elements_update_check_axis_validation)
{
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Constant>(element::i16, axis_shape, 8);

    try
    {
        auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
        FAIL() << "Not detected axis with value out of the range";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Axis value has to be in range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
