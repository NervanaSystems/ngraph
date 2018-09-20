/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifdef NGRAPH_DISTRIBUTED

#include "ngraph/distributed.hpp"
#include <mpi.h>

using namespace ngraph;

ngraph::Distributed::Distributed()
{
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag)
    {
        MPI_Init(NULL, NULL);
    }
}

ngraph::Distributed::~Distributed()
{
    MPI_Finalize();
}

int ngraph::Distributed::get_size() const
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

int ngraph::Distributed::get_rank() const
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}
#endif
