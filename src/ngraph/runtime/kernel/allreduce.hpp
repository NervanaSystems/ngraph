// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#ifdef NGRAPH_DISTRIBUTED

#include <mpi.h>
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace kernel
        {
            template <typename T>
            void allreduce(const T* arg, T* out, const element::Type element_type, int count)
            {
                auto data_type = MPI_FLOAT;

                if (element_type == element::f32)
                {
                    data_type = MPI_FLOAT;
                }
                else if (element_type == element::f64)
                {
                    data_type = MPI_DOUBLE;
                }

                MPI_Allreduce(arg, out, count, data_type, MPI_SUM, MPI_COMM_WORLD);
            }
        }
    }
}

#endif
