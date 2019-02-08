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

namespace ngraph_dist_setup
{
    extern int distributed_comm_size;
    extern int distributed_comm_rank;
}

class DistributedSetup
{
public:
    int get_comm_size();
    int get_comm_rank();
    void set_comm_size(int comm_size);
    void set_comm_rank(int comm_rank);
};
