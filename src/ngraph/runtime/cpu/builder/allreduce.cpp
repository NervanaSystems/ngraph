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

#include "ngraph/op/allreduce.hpp"
#include <mpi.h>
#include "ngraph/runtime/cpu/cpu_builder.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::AllReduce)
            {
                auto& functors = external_function->get_functors();

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                auto count = static_cast<int>(out[0].get_size());
                auto data_type = MPI_FLOAT;

                if (args[0].get_element_type() == element::f32)
                {
                    data_type = MPI_FLOAT;
                }
                else if (args[0].get_element_type() == element::f64)
                {
                    data_type = MPI_DOUBLE;
                }

                auto functor = [&, count, data_type](CPURuntimeContext* ctx) {
                    MPI_Allreduce(
                        arg_tensor, out_tensor, count, data_type, MPI_SUM, MPI_COMM_WORLD);
                };

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(AllReduce);
        }
    }
}
#endif
