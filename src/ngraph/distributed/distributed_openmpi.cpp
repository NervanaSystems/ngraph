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

#ifdef NGRAPH_DISTRIBUTED

#include <mpi.h>

#include "ngraph/log.hpp"
#include "ngraph/distributed/distributed_openmpi.hpp"

using namespace ngraph;

ngraph::distributed::DistributedOpenMPI::DistributedOpenMPI()
{   
    NGRAPH_INFO << "DistributedOpenMPI::DistributedOpenMPI() -- begin ";
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag)
    {
        MPI_Init(NULL, NULL);
    }
    NGRAPH_INFO << "DistributedOpenMPI::DistributedOpenMPI() -- end ";
}

void ngraph::distributed::DistributedOpenMPI::finalize()
{   
    NGRAPH_INFO << "DistributedOpenMPI::finalize() -- begin ";
    MPI_Finalize();
    NGRAPH_INFO << "DistributedOpenMPI::finalize() -- end ";
}

int ngraph::distributed::DistributedOpenMPI::get_size() const
{   
    NGRAPH_INFO << "DistributedOpenMPI::get_size() -- begin ";
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    NGRAPH_INFO << "DistributedOpenMPI::get_size() -- end ";
    return size;
}

int ngraph::distributed::DistributedOpenMPI::get_rank() const
{   
    NGRAPH_INFO << "DistributedOpenMPI::get_rank() -- begin ";
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    NGRAPH_INFO << "DistributedOpenMPI::get_rank() -- end ";
    return rank;
}
#endif
