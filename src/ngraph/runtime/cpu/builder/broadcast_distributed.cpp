//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/broadcast_distributed.hpp"
#include "ngraph/distributed.hpp"
#include "ngraph/log.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::BroadcastDistributed)
            {
                (void)out;
                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto count = static_cast<int>(args[0].get_size());
                auto data_type = args[0].get_element_type();
                auto broadcast = static_cast<const ngraph::op::BroadcastDistributed*>(node);
                auto root_id = broadcast->get_root_id();
                auto functor = [&, count, data_type, arg_buffer_index, root_id](
                    CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {
                    get_distributed_interface()->broadcast(
                        ctx->buffer_data[arg_buffer_index], data_type, count, root_id);
                };
                functors.emplace_back(functor);
            }

            void register_builders_broadcast_distributed_cpp()
            {
                REGISTER_OP_BUILDER(BroadcastDistributed);
            }
        }
    }
}
