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

#include <cstring>

#include "ngraph/op/scatter_add.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/scatter_add.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::ScatterAdd)
            {
                (void)node;
                auto& functors = external_function->get_functors();

                auto inputs_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto indices_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto updates_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (args[1].get_element_type() != element::i64 &&
                    args[1].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }

                if (args[0].get_element_type() != element::f64 &&
                    args[0].get_element_type() != element::f32 &&
                    args[0].get_element_type() != element::u8 &&
                    args[0].get_element_type() != element::i8)
                {
                    throw ngraph_error("Unsupported type in CPU Builder for ScatterAdd");
                }

                bool is_int64 = args[1].get_element_type() == element::i64;
                auto inputs_shape = args[0].get_shape();
                auto indices_shape = args[1].get_shape();
                auto updates_shape = args[2].get_shape();
                auto out_shape = out[0].get_shape();
                auto element_type = args[0].get_element_type();

                if (is_int64 && is_optimized_et(args[0].get_element_type()))
                {
                    if (inputs_shape.size() <= 3 && updates_shape.size() <= 5)
                    {
                        std::function<decltype(runtime::cpu::kernel::scatter_add_i64<float, 2, 2>)>
                            kernel;

                        SELECT_RANK35_ET4(kernel,
                                          args[0].get_element_type(),
                                          inputs_shape.size(),
                                          updates_shape.size(),
                                          runtime::cpu::kernel::scatter_add_i64);

                        auto functor = [&,
                                        kernel,
                                        inputs_shape,
                                        indices_shape,
                                        updates_shape,
                                        inputs_buffer_index,
                                        indices_buffer_index,
                                        updates_buffer_index,
                                        out_buffer_index](CPURuntimeContext* ctx,
                                                          CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[inputs_buffer_index],
                                   ctx->buffer_data[indices_buffer_index],
                                   ctx->buffer_data[updates_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   inputs_shape,
                                   indices_shape,
                                   updates_shape,
                                   ectx->arena);
                        };
                        functors.emplace_back(functor);
                    }
                    else
                    {
                        throw ngraph_error("Unsupported ranks in CPU Builder for ScatterAdd");
                    }
                }
                else if (is_optimized_et(args[0].get_element_type()))
                {
                    if (inputs_shape.size() <= 3 && updates_shape.size() <= 5)
                    {
                        std::function<decltype(runtime::cpu::kernel::scatter_add_i32<float, 2, 2>)>
                            kernel;

                        SELECT_RANK35_ET4(kernel,
                                          args[0].get_element_type(),
                                          inputs_shape.size(),
                                          updates_shape.size(),
                                          runtime::cpu::kernel::scatter_add_i32);

                        auto functor = [&,
                                        kernel,
                                        inputs_shape,
                                        indices_shape,
                                        updates_shape,
                                        inputs_buffer_index,
                                        indices_buffer_index,
                                        updates_buffer_index,
                                        out_buffer_index](CPURuntimeContext* ctx,
                                                          CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[inputs_buffer_index],
                                   ctx->buffer_data[indices_buffer_index],
                                   ctx->buffer_data[updates_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   inputs_shape,
                                   indices_shape,
                                   updates_shape,
                                   ectx->arena);
                        };
                        functors.emplace_back(functor);
                    }
                    else
                    {
                        throw ngraph_error("Unsupported ranks in CPU Builder for ScatterAdd");
                    }
                }
                else if (is_int64)
                {
                    std::function<decltype(runtime::cpu::kernel::ref_scatter_add_i64<float>)>
                        kernel;
                    SELECT_KERNEL(kernel,
                                  args[0].get_element_type(),
                                  runtime::cpu::kernel::ref_scatter_add_i64);

                    auto functor = [&,
                                    kernel,
                                    inputs_shape,
                                    indices_shape,
                                    updates_shape,
                                    out_shape,
                                    inputs_buffer_index,
                                    indices_buffer_index,
                                    updates_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /*ectx*/) {
                        kernel(ctx->buffer_data[inputs_buffer_index],
                               ctx->buffer_data[indices_buffer_index],
                               ctx->buffer_data[updates_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               inputs_shape,
                               indices_shape,
                               updates_shape,
                               out_shape);
                    };
                    functors.emplace_back(functor);
                }

                else
                {
                    std::function<decltype(runtime::cpu::kernel::ref_scatter_add_i32<float>)>
                        kernel;
                    SELECT_KERNEL(kernel,
                                  args[0].get_element_type(),
                                  runtime::cpu::kernel::ref_scatter_add_i32);

                    auto functor = [&,
                                    kernel,
                                    inputs_shape,
                                    indices_shape,
                                    updates_shape,
                                    out_shape,
                                    inputs_buffer_index,
                                    indices_buffer_index,
                                    updates_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /*ectx*/) {
                        kernel(ctx->buffer_data[inputs_buffer_index],
                               ctx->buffer_data[indices_buffer_index],
                               ctx->buffer_data[updates_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               inputs_shape,
                               indices_shape,
                               updates_shape,
                               out_shape);
                    };
                    functors.emplace_back(functor);
                }
            }

            void register_builders_scatter_add_cpp() { REGISTER_OP_BUILDER(ScatterAdd); }
        }
    }
}
