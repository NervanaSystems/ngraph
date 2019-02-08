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

#ifdef NGRAPH_DISTRIBUTED_ENABLE

#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
#include <mlsl.hpp>
#elif NGRAPH_DISTRIBUTED_OMPI_ENABLE
#include <mpi.h>
#endif

#include "ngraph/distributed.hpp"
#include "ngraph/log.hpp"

using namespace ngraph;

ngraph::Distributed::Distributed()
{
    NGRAPH_INFO << "initialize () begin";
#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
    if (!MLSL::Environment::GetEnv().IsInitialized())
    {
        MLSL::Environment::GetEnv().Init(nullptr, nullptr);
        this_init_comm = true;
        NGRAPH_INFO << "initialize () MLSL init ";
    }
#elif NGRAPH_DISTRIBUTED_OMPI_ENABLE
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag)
    {
        MPI_Init(NULL, NULL);
        this_init_comm = true;
        NGRAPH_INFO << "initialize () OpenMPI init";
    }
#else
    throw ngraph_error("Distributed Library not supported/mentioned");
#endif
    NGRAPH_INFO << "initialize () end";
}

ngraph::Distributed::~Distributed()
{
    NGRAPH_INFO << "finalize () begin";
    if (this_init_comm == true)
    {
        finalize();
    }
    else
    {
        NGRAPH_INFO << "finalize () skipped";
    }

    NGRAPH_INFO << "finalize () end";
}

void ngraph::Distributed::finalize()
{
#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
    if (MLSL::Environment::GetEnv().IsInitialized())
    {
        MLSL::Environment::GetEnv().Finalize();
        NGRAPH_INFO << "finalize MLSL here ";
    }
#elif NGRAPH_DISTRIBUTED_OMPI_ENABLE
    int flag = 0;
    MPI_Initialized(&flag);
    if (flag)
    {
        MPI_Finalize();
        NGRAPH_INFO << "finalize OpenMPI here ";
    }
#else
    throw ngraph_error("Distributed Library not supported/mentioned");
#endif
}

int ngraph::Distributed::get_size() const
{
#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
    return static_cast<int>(MLSL::Environment::GetEnv().GetProcessCount());
#elif NGRAPH_DISTRIBUTED_OMPI_ENABLE
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
#else
    throw ngraph_error("Distributed Library not supported/mentioned");
#endif
}

int ngraph::Distributed::get_rank() const
{
#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
    return static_cast<int>(MLSL::Environment::GetEnv().GetProcessIdx());
#elif NGRAPH_DISTRIBUTED_OMPI_ENABLE
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    throw ngraph_error("Distributed Library not supported/mentioned");
#endif
}
#endif
