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

#include "ngraph/op/gather_nd.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::GatherND)
            {
                (void)node;
                auto& functors = external_function->get_functors();
                CPUKernelFunctor functor;

                auto params_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto indices_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                if (args[1].get_element_type() != element::i64 &&
                    args[1].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }
                bool is_int64 = args[1].get_element_type() == element::i64;
                auto params_shape = args[0].get_shape();
                auto indices_shape = args[1].get_shape();
                auto out_shape = out[0].get_shape();
                auto element_type = args[0].get_element_type();
                if (element_type == element::f32)
                {
                    if (is_int64)
                    {
                        functor = [&,
                                   params_shape,
                                   indices_shape,
                                   out_shape,
                                   params_buffer_index,
                                   indices_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* /* ectx */) {
                            ngraph::runtime::reference::gather_nd<float, int64_t>(
                                static_cast<float*>(ctx->buffer_data[params_buffer_index]),
                                static_cast<int64_t*>(ctx->buffer_data[indices_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                params_shape,
                                indices_shape,
                                out_shape);
                        };
                    }
                    else
                    {
                        functor = [&,
                                   params_shape,
                                   indices_shape,
                                   out_shape,
                                   params_buffer_index,
                                   indices_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* /* ectx */) {
                            ngraph::runtime::reference::gather_nd<float, int32_t>(
                                static_cast<float*>(ctx->buffer_data[params_buffer_index]),
                                static_cast<int32_t*>(ctx->buffer_data[indices_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                params_shape,
                                indices_shape,
                                out_shape);
                        };
                    }
                }
                else if (element_type == element::f64)
                {
                    if (is_int64)
                    {
                        functor = [&,
                                   params_shape,
                                   indices_shape,
                                   out_shape,
                                   params_buffer_index,
                                   indices_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* /* ectx */) {
                            ngraph::runtime::reference::gather_nd<double, int64_t>(
                                static_cast<double*>(ctx->buffer_data[params_buffer_index]),
                                static_cast<int64_t*>(ctx->buffer_data[indices_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                params_shape,
                                indices_shape,
                                out_shape);
                        };
                    }
                    else
                    {
                        functor = [&,
                                   params_shape,
                                   indices_shape,
                                   out_shape,
                                   params_buffer_index,
                                   indices_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* /* ectx */) {
                            ngraph::runtime::reference::gather_nd<double, int32_t>(
                                static_cast<double*>(ctx->buffer_data[params_buffer_index]),
                                static_cast<int32_t*>(ctx->buffer_data[indices_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                params_shape,
                                indices_shape,
                                out_shape);
                        };
                    }
                }
                else
                {
                    throw ngraph_error("Unsupported type in CPU Builder for GatherND");
                }

                functors.emplace_back(functor);
            }

            void register_builders_gather_nd_cpp() { REGISTER_OP_BUILDER(GatherND); }
        }
    }
}
