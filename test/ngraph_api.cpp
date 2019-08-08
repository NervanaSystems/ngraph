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

using namespace ngraph;
using namespace std;

TEST(ngraph_api, parse_version)
{
    size_t major;
    size_t minor;
    size_t patch;
    const char* extra;

    {
        string version = "0.25.1-rc.0+7c32240";
        parse_version_string(version, major, minor, patch, &extra);
        EXPECT_EQ(0, major);
        EXPECT_EQ(25, minor);
        EXPECT_EQ(1, patch);
        EXPECT_STREQ(extra, "-rc.0+7c32240");
        NGRAPH_INFO << major;
        NGRAPH_INFO << minor;
        NGRAPH_INFO << patch;
        NGRAPH_INFO << extra;
    }
}

TEST(ngraph_api, version)
{
    size_t major;
    size_t minor;
    size_t patch;
    const char* extra;
    get_ngraph_version(major, minor, patch, &extra);
    NGRAPH_INFO << major;
    NGRAPH_INFO << minor;
    NGRAPH_INFO << patch;
    NGRAPH_INFO << extra;
}
