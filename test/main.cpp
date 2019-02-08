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

#include "distributed_setup.hpp"
#include "gtest/gtest.h"
#include "ngraph/log.hpp"

using namespace std;

#ifdef NGRAPH_DISTRIBUTED_ENABLE
#include "ngraph/distributed.hpp"
#endif

int main(int argc, char** argv)
{
#ifdef NGRAPH_DISTRIBUTED_ENABLE
    auto dist = new ngraph::Distributed;
    bool delete_instance = true;
    DistributedSetup distributed_setup;
    distributed_setup.set_comm_size(dist->get_size());
    distributed_setup.set_comm_rank(dist->get_rank());
    NGRAPH_INFO << "initialize instance in main() ";
    if (dist->get_size() == 1)
    {
        NGRAPH_INFO << "delete instance in main get_size()==1 ";
        delete dist;
        delete_instance = false;
    }
#endif
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
    int rc = RUN_ALL_TESTS();

#ifdef NGRAPH_DISTRIBUTED_ENABLE

    NGRAPH_INFO << "delete instance in main end ";
    if (delete_instance)
    {
        delete dist;
    }
#endif
    return rc;
}
