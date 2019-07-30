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

#include "ngraph/op/gather.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/gather.hpp"
#include "ngraph/runtime/reference/gather.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace
            {
                template <typename T>
                CPUKernelFunctor prepare_functor(const Node* node,
                                                 const vector<TensorViewWrapper>& args,
                                                 const vector<TensorViewWrapper>& out,
                                                 CPU_ExternalFunction* external_function)
                {
                    const ngraph::op::Gather* gather = static_cast<const ngraph::op::Gather*>(node);
                    auto params_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto indices_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    bool is_int64 = args[1].get_element_type() == element::i64;
                    auto axis = gather->get_axis();
                    auto params_shape = args[0].get_shape();
                    auto indices_shape = args[1].get_shape();
                    auto out_shape = out[0].get_shape();

                    if (is_int64)
                    {
                        if ((args[0].get_element_type() == element::f32 ||
                             args[0].get_element_type() == element::f64 ||
                             args[0].get_element_type() == element::u8 ||
                             args[0].get_element_type() == element::i8) &&
                            params_shape.size() <= 3 && out_shape.size() <= 5)
                        {
                            std::function<decltype(runtime::cpu::kernel::gather_i64<float, 2, 2>)>
                                kernel;

                            SELECT_KERNEL_BY_2RANKS(kernel,
                                                    args[0].get_element_type(),
                                                    params_shape.size(),
                                                    out_shape.size(),
                                                    runtime::cpu::kernel::gather_i64);

                            return [&,
                                    kernel,
                                    params_shape,
                                    indices_shape,
                                    out_shape,
                                    axis,
                                    params_buffer_index,
                                    indices_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                                kernel(ctx->buffer_data[params_buffer_index],
                                       ctx->buffer_data[indices_buffer_index],
                                       ctx->buffer_data[out_buffer_index],
                                       params_shape,
                                       indices_shape,
                                       out_shape,
                                       axis,
                                       ectx->arena);
                            };
                        }
                        else
                        {
                            return [&,
                                    params_shape,
                                    indices_shape,
                                    out_shape,
                                    axis,
                                    params_buffer_index,
                                    indices_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                                ngraph::runtime::reference::gather<T, int64_t>(
                                    static_cast<T*>(ctx->buffer_data[params_buffer_index]),
                                    static_cast<int64_t*>(ctx->buffer_data[indices_buffer_index]),
                                    static_cast<T*>(ctx->buffer_data[out_buffer_index]),
                                    params_shape,
                                    indices_shape,
                                    out_shape,
                                    axis);
                            };
                        }
                    }

                    else
                    {
                        if ((args[0].get_element_type() == element::f32 ||
                             args[0].get_element_type() == element::f64 ||
                             args[0].get_element_type() == element::u8 ||
                             args[0].get_element_type() == element::i8) &&
                            params_shape.size() <= 3 && out_shape.size() <= 5)
                        {
                            std::function<decltype(runtime::cpu::kernel::gather_i32<float, 2, 2>)>
                                kernel;

                            SELECT_KERNEL_BY_2RANKS(kernel,
                                                    args[0].get_element_type(),
                                                    params_shape.size(),
                                                    out_shape.size(),
                                                    runtime::cpu::kernel::gather_i32);

                            return [&,
                                    kernel,
                                    params_shape,
                                    indices_shape,
                                    out_shape,
                                    axis,
                                    params_buffer_index,
                                    indices_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                                kernel(ctx->buffer_data[params_buffer_index],
                                       ctx->buffer_data[indices_buffer_index],
                                       ctx->buffer_data[out_buffer_index],
                                       params_shape,
                                       indices_shape,
                                       out_shape,
                                       axis,
                                       ectx->arena);
                            };
                        }
                        else
                        {
                            return [&,
                                    params_shape,
                                    indices_shape,
                                    out_shape,
                                    axis,
                                    params_buffer_index,
                                    indices_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                                ngraph::runtime::reference::gather<T, int32_t>(
                                    static_cast<T*>(ctx->buffer_data[params_buffer_index]),
                                    static_cast<int32_t*>(ctx->buffer_data[indices_buffer_index]),
                                    static_cast<T*>(ctx->buffer_data[out_buffer_index]),
                                    params_shape,
                                    indices_shape,
                                    out_shape,
                                    axis);
                            };
                        }
                    }
                }
            } // namespace

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Gather)
            {
                auto& functors = external_function->get_functors();
                CPUKernelFunctor functor;
                if (args[1].get_element_type() != element::i64 &&
                    args[1].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }
                auto element_type = args[0].get_element_type();
                if (element_type == element::f32)
                {
                    functor = prepare_functor<float>(node, args, out, external_function);
                }
                else if (element_type == element::f64)
                {
                    functor = prepare_functor<double>(node, args, out, external_function);
                }
                else if (element_type == element::i8)
                {
                    functor = prepare_functor<int8_t>(node, args, out, external_function);
                }
                else if (element_type == element::i16)
                {
                    functor = prepare_functor<int16_t>(node, args, out, external_function);
                }
                else if (element_type == element::i32)
                {
                    functor = prepare_functor<int32_t>(node, args, out, external_function);
                }
                else if (element_type == element::i64)
                {
                    functor = prepare_functor<int64_t>(node, args, out, external_function);
                }
                else if (element_type == element::u8)
                {
                    functor = prepare_functor<uint8_t>(node, args, out, external_function);
                }
                else if (element_type == element::u16)
                {
                    functor = prepare_functor<uint16_t>(node, args, out, external_function);
                }
                else if (element_type == element::u32)
                {
                    functor = prepare_functor<uint32_t>(node, args, out, external_function);
                }
                else if (element_type == element::u64)
                {
                    functor = prepare_functor<uint64_t>(node, args, out, external_function);
                }
                else if (element_type == element::boolean)
                {
                    functor = prepare_functor<char>(node, args, out, external_function);
                }
                else
                {
                    throw ngraph_error("Unsupported type in CPU Builder for Gather");
                }

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Gather);
        } // namespace cpu
    }     // namespace runtime
} // namespace ngraph
