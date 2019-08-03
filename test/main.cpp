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

#include <chrono>
#include <iostream>

#include "gtest/gtest.h"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"

using namespace std;

static void configure_static_backends()
{
#ifdef NGRAPH_INTERPRETER_STATIC_LIB_ENABLE
    ngraph::runtime::BackendManager::register_backend(
        "INTERPRETER", ngraph::runtime::interpreter::get_backend_constructor_pointer());
#endif
}

int main(int argc, char** argv)
{
    configure_static_backends();
    const char* exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    argv_vector.push_back(const_cast<char*>(exclude));
    for (int i = 1; i < argc; i++)
    {
        argv_vector.push_back(argv[i]);
    }
    argc++;

    ::testing::InitGoogleTest(&argc, argv_vector.data());
    auto start = std::chrono::system_clock::now();
    int rc = RUN_ALL_TESTS();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - start);
    NGRAPH_DEBUG_PRINT("[MAIN] Tests finished: Time: %d ms Exit code: %d", elapsed.count(), rc);

    return rc;
}
