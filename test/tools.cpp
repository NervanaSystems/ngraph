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

#include <gtest/gtest.h>
#include <sstream>

#include "misc.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"

using namespace ngraph;
using namespace std;

TEST(tools, nbench_functional)
{
    const string model = "mxnet/mnist_mlp_forward.json";
    const string model_path = file_util::path_join(SERIALIZED_ZOO, model);

    stringstream ss;

    ss << NBENCH_PATH << " -f " << model_path << " -b INTERPRETER -i 2 -w 2";
    auto cmd = ss.str();
    auto f = port_open(cmd.c_str(), "r");
    if (f)
    {
        stringstream str;
        char buffer[256];
        while (!feof(f))
        {
            size_t count = fread(buffer, 1, sizeof(buffer), f);
            string s = string(buffer, count);
            str << s;
        }
        string output = str.str();
        auto status = port_close(f);
        ASSERT_EQ(status, 0) << output;
    }
    else
    {
        FAIL();
    }
}
