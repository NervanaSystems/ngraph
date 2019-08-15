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

#include "ngraph/runtime/cpu/kernel/one_hot.hpp"
#include "ngraph/op/one_hot.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::OneHot)
            {
                auto oh = static_cast<const ngraph::op::OneHot*>(node);
                auto one_hot_axis = oh->get_one_hot_axis();
                auto arg_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();
                auto out_strides = out[0].get_strides();
                auto arg_rank = arg_shape.size();

                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (arg_rank == 0)
                {
                    std::function<decltype(runtime::cpu::kernel::one_hot_rank_0<float>)> kernel;
                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::one_hot_rank_0);
                    auto functor =
                        [&, kernel, out_shape, one_hot_axis, arg_buffer_index, out_buffer_index](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   out_shape,
                                   one_hot_axis,
                                   ectx->arena);
                        };

                    functors.emplace_back(functor);
                }
                else if (arg_rank == 1)
                {
                    std::function<decltype(runtime::cpu::kernel::one_hot_rank_1<float>)> kernel;
                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::one_hot_rank_1);
                    auto functor = [&,
                                    kernel,
                                    arg_shape,
                                    out_shape,
                                    one_hot_axis,
                                    arg_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg_shape,
                               out_shape,
                               one_hot_axis,
                               ectx->arena);
                    };

                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::one_hot_rank_2_or_more<float>)>
                        kernel;
                    SELECT_KERNEL(kernel,
                                  out[0].get_element_type(),
                                  runtime::cpu::kernel::one_hot_rank_2_or_more);
                    auto functor = [&,
                                    kernel,
                                    arg_shape,
                                    out_shape,
                                    one_hot_axis,
                                    arg_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg_shape,
                               out_shape,
                               one_hot_axis);
                    };

                    functors.emplace_back(functor);
                }
            }

            REGISTER_OP_BUILDER(OneHot);
#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
            void register_builders_one_hot_cpp() {}
#endif
        }
    }
}
