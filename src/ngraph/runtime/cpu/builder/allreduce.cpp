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

#include "ngraph/op/allreduce.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::AllReduce)
            {
                static int call_seq = 0;

                auto& functors = external_function->get_functors();
                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto count = static_cast<int>(out[0].get_size());
                auto data_type = args[0].get_element_type();
                const ngraph::op::AllReduce* allreduce =
                    static_cast<const ngraph::op::AllReduce*>(node);
                auto reduce_type = allreduce->get_reduce_type();

                auto external_function_name = external_function->get_function_name();
                NGRAPH_DEBUG_PRINT(
                    "AllReduce Queued[%d]: Function: %s Node: %s %s Size: "
                    "%d",
                    call_seq,
                    external_function_name.c_str(),
                    node->get_name().c_str(),
                    // if provenance_tags is set in nGraph once and only once, it will print the tag
                    // name otherwise, it will print the get_friendly_name
                    node->get_provenance_tags().size() == 1
                        ? (*(node->get_provenance_tags()).begin()).c_str()
                        : node->get_friendly_name().c_str(),
                    count);

                auto functor =
                    [&, count, reduce_type, data_type, arg_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {
                        get_distributed_interface()->all_reduce(ctx->buffer_data[arg_buffer_index],
                                                                ctx->buffer_data[out_buffer_index],
                                                                data_type,
                                                                reduce_type,
                                                                count);
                    };
                functors.emplace_back(functor);
            }

            void register_builders_allreduce_cpp() { REGISTER_OP_BUILDER(AllReduce); }
        }
    }
}
