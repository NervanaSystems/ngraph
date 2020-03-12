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

using namespace std;
using namespace ngraph;

TEST(type_prop_layers, roi_align_basic_shape_inference)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 5, 5});
    const auto rois = make_shared<op::Parameter>(element::f32, Shape{7, 4});
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{7});
    const auto op = make_shared<op::ROIAlign>(data, rois, batch_indices, 2, 2, 1, 1.0f, "avg");
    ASSERT_EQ(op->get_shape(), (Shape{7, 3, 2, 2}));
}
