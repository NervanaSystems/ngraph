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

#include "ngraph/op/topk.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/reference/topk.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::TopK)
            {
                auto& functors = external_function->get_functors();
                const ngraph::op::TopK* topk = static_cast<const ngraph::op::TopK*>(node);
                CPUKernelFunctor functor;

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_indices_buffer_index =
                    external_function->get_buffer_index(out[0].get_name());
                auto out_values_buffer_index =
                    external_function->get_buffer_index(out[1].get_name());
                if (out[0].get_element_type() != element::i64 &&
                    out[0].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }
                bool is_int64 = out[0].get_element_type() == element::i64;
                auto axis = topk->get_top_k_axis();
                auto in_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();
                auto k = topk->get_k();
                auto compute_max = topk->get_compute_max();

                auto element_type = args[0].get_element_type();
                if (element_type == element::f32)
                {
                    if (is_int64)
                    {
                        functor = [&,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   k,
                                   compute_max,
                                   arg_buffer_index,
                                   out_indices_buffer_index,
                                   out_values_buffer_index](CPURuntimeContext* ctx,
                                                            CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<float, int64_t>(
                                static_cast<float*>(ctx->buffer_data[arg_buffer_index]),
                                static_cast<int64_t*>(ctx->buffer_data[out_indices_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[out_values_buffer_index]),
                                in_shape,
                                out_shape,
                                axis,
                                k,
                                compute_max);
                        };
                    }
                    else
                    {
                        functor = [&,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   k,
                                   compute_max,
                                   arg_buffer_index,
                                   out_indices_buffer_index,
                                   out_values_buffer_index](CPURuntimeContext* ctx,
                                                            CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<float, int32_t>(
                                static_cast<float*>(ctx->buffer_data[arg_buffer_index]),
                                static_cast<int32_t*>(ctx->buffer_data[out_indices_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[out_values_buffer_index]),
                                in_shape,
                                out_shape,
                                axis,
                                k,
                                compute_max);
                        };
                    }
                }
                else if (element_type == element::f64)
                {
                    if (is_int64)
                    {
                        functor = [&,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   k,
                                   compute_max,
                                   arg_buffer_index,
                                   out_indices_buffer_index,
                                   out_values_buffer_index](CPURuntimeContext* ctx,
                                                            CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<double, int64_t>(
                                static_cast<double*>(ctx->buffer_data[arg_buffer_index]),
                                static_cast<int64_t*>(ctx->buffer_data[out_indices_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[out_values_buffer_index]),
                                in_shape,
                                out_shape,
                                axis,
                                k,
                                compute_max);
                        };
                    }
                    else
                    {
                        functor = [&,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   k,
                                   compute_max,
                                   arg_buffer_index,
                                   out_indices_buffer_index,
                                   out_values_buffer_index](CPURuntimeContext* ctx,
                                                            CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<double, int32_t>(
                                static_cast<double*>(ctx->buffer_data[arg_buffer_index]),
                                static_cast<int32_t*>(ctx->buffer_data[out_indices_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[out_values_buffer_index]),
                                in_shape,
                                out_shape,
                                axis,
                                k,
                                compute_max);
                        };
                    }
                }
                else if (element_type == element::i32)
                {
                    if (is_int64)
                    {
                        functor = [&,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   k,
                                   compute_max,
                                   arg_buffer_index,
                                   out_indices_buffer_index,
                                   out_values_buffer_index](CPURuntimeContext* ctx,
                                                            CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<int32_t, int64_t>(
                                static_cast<int32_t*>(ctx->buffer_data[arg_buffer_index]),
                                static_cast<int64_t*>(ctx->buffer_data[out_indices_buffer_index]),
                                static_cast<int32_t*>(ctx->buffer_data[out_values_buffer_index]),
                                in_shape,
                                out_shape,
                                axis,
                                k,
                                compute_max);
                        };
                    }
                    else
                    {
                        functor = [&,
                                   in_shape,
                                   out_shape,
                                   axis,
                                   k,
                                   compute_max,
                                   arg_buffer_index,
                                   out_indices_buffer_index,
                                   out_values_buffer_index](CPURuntimeContext* ctx,
                                                            CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<int32_t, int32_t>(
                                static_cast<int32_t*>(ctx->buffer_data[arg_buffer_index]),
                                static_cast<int32_t*>(ctx->buffer_data[out_indices_buffer_index]),
                                static_cast<int32_t*>(ctx->buffer_data[out_values_buffer_index]),
                                in_shape,
                                out_shape,
                                axis,
                                k,
                                compute_max);
                        };
                    }
                }
                else
                {
                    throw ngraph_error("Unsupported type (" + element_type.get_type_name() +
                                       ") in CPU Builder for TopK");
                }

                functors.emplace_back(functor);
            }

            void register_builders_topk_cpp() { REGISTER_OP_BUILDER(TopK); }
        }
    }
}
