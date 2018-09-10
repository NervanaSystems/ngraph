//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

using namespace std;

#ifdef NGRAPH_DISTRIBUTED
#include <mpi.h>

class MpiEnvironment : public ::testing::Environment
{
protected:
    virtual void SetUp()
    {
        int flag = 0;
        MPI_Initialized(&flag);
        if (!flag)
        {
            MPI::Init();
        }
    }
    virtual void TearDown() { MPI::Finalize(); }
    virtual ~MpiEnvironment() {}
};

#endif

int main(int argc, char** argv)
{
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
#ifdef NGRAPH_DISTRIBUTED
    ::testing::AddGlobalTestEnvironment(new MpiEnvironment);
#endif
    int rc = RUN_ALL_TESTS();

    return rc;
}
