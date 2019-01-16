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

#include <mlsl.hpp>

#include "ngraph/distributed.hpp"

using namespace ngraph;

// use the previous MPI implementation
ngraph::Distributed_MPI::Distributed()
{
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag)
    {
        MPI_Init(NULL, NULL);
    }
}

ngraph::Distributed_MPI::~Distributed()
{
    MPI_Finalize();
}

size_t ngraph::Distributed_MPI::get_process_count()
{
    size_t size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

   
size_t ngraph::Distributed_MPI::get_process_id()
{
    size_t process_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    return process_id;
}

ngraph::Distributed_MLSL::Distributed()
{
    if (!MLSL::Environment::GetEnv().IsInitialized())
    {
        MLSL::Environment::GetEnv().Init(nullptr, nullptr);
    }
}

ngraph::Distributed_MLSL::~Distributed()
{
    if (MLSL::Environment::GetEnv().IsInitialized())
    {
        MLSL::Environment::GetEnv().Finalize();
    }
}

size_t ngraph::Distributed_MLSL::get_process_count() const
{
    return MLSL::Environment::GetEnv().GetProcessCount();
}

size_t ngraph::Distributed_MLSL::get_process_id() const
{
    return MLSL::Environment::GetEnv().GetProcessIdx();
}
#endif
