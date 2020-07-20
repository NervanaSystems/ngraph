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

#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace std;

TEST(typeinfo, validate)
{
    // This test is designed to validate that each node's type_info is set correctly so that the
    // class name exactly matches the type_info name and that the version of the op matches
    // the value define in op_tbl
    //
    // Each entry expands like this
    // EXPECT_STREQ(ngraph::op::v0::Abs::type_info.name, "Abs");
    // EXPECT_EQ(ngraph::op::v0::Abs::type_info.version, 0);
#define NGRAPH_OP(NAME, VER)                                                                       \
    EXPECT_STREQ(ngraph::op::v##VER::NAME::type_info.name, #NAME);                                 \
    EXPECT_EQ(ngraph::op::v##VER::NAME::type_info.version, VER);
#include "ngraph/op_version_tbl.hpp"
#undef NGRAHP_OP
}
