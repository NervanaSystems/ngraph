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

#pragma once

#ifdef IN_NGRAPH_LIBRARY
#error("distributed.hpp is for external use only")
#endif

#include <mpi.h>

namespace ngraph
{
    class Distributed
    {
    public:
        Distributed()
        {
            int flag = 0;
            MPI_Initialized(&flag);
            if (!flag)
            {
                MPI_Init(NULL, NULL);
            }
        }

        ~Distributed() { MPI_Finalize(); }
        int get_size()
        {
            int size;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return size;
        }

        int get_rank()
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank;
        }
    };
}
