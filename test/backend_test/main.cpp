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

#include <iostream>

#include "backend_test/backend_test.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 2 && argc != 3)
    {
        std::cerr << "Syntax: " << argv[0] << " [GTEST_OPTIONS] <BACKEND_NAME> [MANIFEST_FILE]"
                  << std::endl;
        return 1;
    }

    BackendTest::set_backend_under_test(argv[1]);

    if (argc == 3)
    {
        BackendTest::load_manifest(argv[2]);
    }

    int rc = RUN_ALL_TESTS();
    return rc;
}
