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

#include "ngraph/log.hpp"
#include "ngraph/op/allreduce.hpp"
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
                static int call_seq = 0;

                auto& functors = external_function->get_functors();
                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                auto count = static_cast<int>(out[0].get_size());

                auto external_function_name = external_function->get_function_name();
                NGRAPH_DEBUG_PRINT(
                    "AllReduce Queued[%d]: Function: %s Node: %s %s Size: "
                    "%d",
                    call_seq,
                    external_function_name.c_str(),
                    node->get_name().c_str(),
                    node->get_friendly_name().c_str(),
                    count);

#ifdef NGRAPH_DISTRIBUTED_MLSL_ENABLE
                auto data_type = MLSL::DT_FLOAT;

                if (args[0].get_element_type() == element::f32)
                {
                    data_type = MLSL::DT_FLOAT;
                }
                else if (args[0].get_element_type() == element::f64)
                {
                    data_type = MLSL::DT_DOUBLE;
                }

                auto functor = [&, count, data_type](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {
                    MLSL::CommReq* req = ctx->mlsl_dist->AllReduce(
                        arg_tensor, out_tensor, count, data_type, MLSL::RT_SUM, MLSL::GT_DATA);
                    ctx->mlsl_env->Wait(req);
                };
#elif NGRAPH_DISTRIBUTED_OMPI_ENABLE
                auto data_type = MPI_FLOAT;

                if (args[0].get_element_type() == element::f32)
                {
                    data_type = MPI_FLOAT;
                }
                else if (args[0].get_element_type() == element::f64)
                {
                    data_type = MPI_DOUBLE;
                }

                auto node_friendly_name = node->get_friendly_name();
                auto node_name = node->get_name();
                auto func_name = external_function->get_function_name();
                int id = call_seq;
                call_seq++;

                auto functor = [&, id, count, data_type, func_name, node_friendly_name, node_name](
                    CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                    NGRAPH_DEBUG_PRINT("AllReduce Execute[%d]: Function: %s  Node: %s %s Size: %d",
                                       id,
                                       func_name.c_str(),
                                       node_name.c_str(),
                                       node_friendly_name.c_str(),
                                       count);
                    MPI_Allreduce(
                        arg_tensor, out_tensor, count, data_type, MPI_SUM, MPI_COMM_WORLD);
                };
#else
                throw ngraph_error("Distributed Library not supported/mentioned");
#endif
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(AllReduce);
        }
    }
}
#endif
