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

#ifdef NGRAPH_DISTRIBUTED

// #ifdef NGRAPH_CPU_ENABLE
// #include <mlsl.hpp>
// #endif

#include <mpi.h>

#include "ngraph/distributed.hpp"

using namespace ngraph;

int ngraph::distributed::distributed_get_rank(){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
    return rank;
}

ngraph::Distributed::Distributed()
{
// #ifdef NGRAPH_CPU_ENABLE
//     if (!MLSL::Environment::GetEnv().IsInitialized())
//     {
//         MLSL::Environment::GetEnv().Init(nullptr, nullptr);
//     }
// #endif

    int flag = 0;
    MPI_Initialized(&flag);	
    if (!flag)
    {
        MPI_Init(NULL, NULL);
    }

}

ngraph::Distributed::~Distributed()
{
// #ifdef NGRAPH_CPU_ENABLE
//     if (MLSL::Environment::GetEnv().IsInitialized())
//     {
//         MLSL::Environment::GetEnv().Finalize();
//     }
// #endif
    // MPI_Finalize();
}

void ngraph::Distributed::initialize()
{
// #ifdef NGRAPH_CPU_ENABLE
//     if (!MLSL::Environment::GetEnv().IsInitialized())
//     {
//         MLSL::Environment::GetEnv().Init(nullptr, nullptr);
//     }
// #endif

    int flag = 0;
    MPI_Initialized(&flag);	
    if (!flag)
    {
        MPI_Init(NULL, NULL);
    }

}

void ngraph::Distributed::finalize()
{
// #ifdef NGRAPH_CPU_ENABLE
//     if (MLSL::Environment::GetEnv().IsInitialized())
//     {
//         MLSL::Environment::GetEnv().Finalize();
//     }
// #endif
    MPI_Finalize();
}

int ngraph::Distributed::get_size() const
{
// #ifdef NGRAPH_CPU_ENABLE
//     return MLSL::Environment::GetEnv().GetProcessCount();
// #endif
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);	
    return size;
}

int ngraph::Distributed::get_rank() const
{
// #ifdef NGRAPH_CPU_ENABLE
//     return MLSL::Environment::GetEnv().GetProcessIdx();
// #endif
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
    return rank;
}
#endif
