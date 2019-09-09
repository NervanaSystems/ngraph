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

#include <vector>

#include "ngraph/op/add.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/add.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Add)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto sum_pd = mkldnn_emitter->get_elementwise_add_desc(node);
                    QUERY_SCRATCHPAD(sum, sum_pd);

                    // Add needs 4 primitives: input0, input1, result, and sum.
                    size_t add_index = mkldnn_emitter->reserve_primitive_space(4);
                    auto& deps = mkldnn_emitter->get_primitive_deps(add_index);

                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto functor = [&,
                                    sum_pd,
                                    add_index,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_elementwise_add(ctx->mkldnn_memories,
                                                                  ctx->mkldnn_primitives,
                                                                  ctx->mkldnn_scratchpad_mds,
                                                                  sum_pd,
                                                                  deps,
                                                                  add_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx, add_index, deps, cpu::mkldnn_utils::OpType::ADD);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::add);
                }
            }

            void register_builders_add_cpp() { REGISTER_OP_BUILDER(Add); }
        }
    }
}
