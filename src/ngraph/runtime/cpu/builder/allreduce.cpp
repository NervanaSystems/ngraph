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

#include "ngraph/op/allreduce.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"

#include <cstdarg>
#include <iomanip>
#include <locale>
#include <sys/time.h>
#include <unistd.h> /*for CLOCK_REALTIME? */

#include "ngraph/distributed.hpp"

std::string get_timestamp()
{
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of nanoseconds for the current second
    // (remainder after division into seconds)
    auto ns = duration_cast<nanoseconds>(now.time_since_epoch()) % 1000000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream timestamp;
    timestamp << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
    timestamp << '.' << std::setfill('0') << std::setw(3) << ns.count();

    return timestamp.str();
}

// This function will be executed only once during startup (loading of the DSO)
static bool CheckLoggingLevel()
{
    if (std::getenv("NGRAPH_DIST_DISABLE_LOGGING") != nullptr)
    {
        return true;
    }
    return false;
}
bool DISABLE_LOGGING = CheckLoggingLevel();

inline void LogPrintf(const char* fmt, ...)
{
    va_list args1;
    va_start(args1, fmt);
    va_list args2;
    va_copy(args2, args1);

    std::vector<char> buf(1 + std::vsnprintf(nullptr, 0, fmt, args1));
    va_end(args1);
    std::vsnprintf(buf.data(), buf.size(), fmt, args2);
    va_end(args2);

    ngraph::Distributed dist;
    std::printf("%s [RANK: %d]: %s\n", get_timestamp().c_str(), dist.get_rank(), buf.data());
}
#define NGRAPH_DIST_DEBUG(fmt, ...)                                                                \
    if (!DISABLE_LOGGING)                                                                          \
    LogPrintf(fmt, ##__VA_ARGS__)

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
                NGRAPH_DIST_DEBUG(
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
                    NGRAPH_DIST_DEBUG("AllReduce Execute[%d]: Function: %s  Node: %s %s Size: %d",
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
