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

#include <cstring>

#include "ngraph/op/argmax.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/argmax.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::ArgMax)
            {
                auto& functors = external_function->get_functors();

                const ngraph::op::ArgMax* argmax = static_cast<const ngraph::op::ArgMax*>(node);
                CPUKernelFunctor functor;

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                if (out[0].get_element_type() != element::i64 &&
                    out[0].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }
                bool is_int64 = out[0].get_element_type() == element::i64;
                auto axis = argmax->get_reduction_axis();
                auto in_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto element_type = args[0].get_element_type();
                if (element_type == element::f32)
                {
                    if (is_int64)
                    {
                        std::function<decltype(runtime::cpu::kernel::argmax<float, int64_t, 1>)>
                            kernel;

                        SELECT_RANK2(
                            kernel, float, int64_t, in_shape.size(), runtime::cpu::kernel::argmax)

                        functor = [&,
                                   kernel,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   arg_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   in_shape,
                                   out_shape,
                                   axis,
                                   ectx->arena);
                        };
                    }
                    else
                    {
                        std::function<decltype(runtime::cpu::kernel::argmax<float, int, 1>)> kernel;

                        SELECT_RANK2(
                            kernel, float, int, in_shape.size(), runtime::cpu::kernel::argmax)

                        functor = [&,
                                   kernel,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   arg_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   in_shape,
                                   out_shape,
                                   axis,
                                   ectx->arena);
                        };
                    }
                }
                else if (element_type == element::f64)
                {
                    if (is_int64)
                    {
                        std::function<decltype(runtime::cpu::kernel::argmax<double, int64_t, 1>)>
                            kernel;

                        SELECT_RANK2(
                            kernel, double, int64_t, in_shape.size(), runtime::cpu::kernel::argmax)

                        functor = [&,
                                   kernel,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   arg_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   in_shape,
                                   out_shape,
                                   axis,
                                   ectx->arena);
                        };
                    }
                    else
                    {
                        std::function<decltype(runtime::cpu::kernel::argmax<double, int, 1>)>
                            kernel;

                        SELECT_RANK2(
                            kernel, double, int, in_shape.size(), runtime::cpu::kernel::argmax)

                        functor = [&,
                                   kernel,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   arg_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   in_shape,
                                   out_shape,
                                   axis,
                                   ectx->arena);
                        };
                    }
                }
                else if (element_type == element::i32)
                {
                    if (is_int64)
                    {
                        std::function<decltype(runtime::cpu::kernel::argmax<int, int64_t, 1>)>
                            kernel;

                        SELECT_RANK2(
                            kernel, int, int64_t, in_shape.size(), runtime::cpu::kernel::argmax)

                        functor = [&,
                                   kernel,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   arg_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   in_shape,
                                   out_shape,
                                   axis,
                                   ectx->arena);
                        };
                    }
                    else
                    {
                        std::function<decltype(runtime::cpu::kernel::argmax<int, int, 1>)> kernel;

                        SELECT_RANK2(
                            kernel, int, int, in_shape.size(), runtime::cpu::kernel::argmax)

                        functor = [&,
                                   kernel,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   arg_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   in_shape,
                                   out_shape,
                                   axis,
                                   ectx->arena);
                        };
                    }
                }
                else
                {
                    throw ngraph_error("Unsupported type in CPU Builder for ArgMax");
                }

                functors.emplace_back(functor);
            }

            void register_builders_argmax_cpp() { REGISTER_OP_BUILDER(ArgMax); }
        }
    }
}
