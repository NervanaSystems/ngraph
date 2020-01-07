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

#include <gtest/gtest.h>
#include <vector>

#include "ngraph/frontend/onnxifi/span.hpp"

TEST(onnxifi, span)
{
    using namespace ngraph::onnxifi;

    std::vector<float> floats{0.f, 0.25f, 0.5f, 1.f, 2.f, 3.f, 4.f, 5.5f};
    char* buffer{reinterpret_cast<char*>(floats.data())};
    Span<float> span{buffer, floats.size()};
    for (std::size_t index{0}; index < span.size(); ++index)
    {
        EXPECT_EQ(span.at(index), floats.at(index));
    }
}
