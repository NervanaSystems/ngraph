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

#include <cstdint>
#include <cstring>

#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/reference/embedding_lookup.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::EmbeddingLookup)
            {
                auto& functors = external_function->get_functors();

                CPUKernelFunctor functor;
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                if (out[0].get_element_type() != element::f32 &&
                    out[0].get_element_type() != element::f64)
                {
                    throw ngraph_error("Unsupported output element type");
                }
                auto in_shape = args[1].get_shape();
                size_t element_count = shape_size(args[0].get_shape());
                auto out_shape = out[0].get_shape();
                auto element_type = out[0].get_element_type();
                auto index_element_type = args[0].get_element_type();
                if (element_type == element::f32)
                {
                    if (index_element_type == element::f32)
                    {
                        functor = [&,
                                   in_shape,
                                   element_count,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<float, float>(
                                static_cast<float*>(ctx->buffer_data[arg0_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                in_shape);
                        };
                    }
                    else if (index_element_type == element::i32)
                    {
                        functor = [&,
                                   in_shape,
                                   element_count,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<float, int>(
                                static_cast<int*>(ctx->buffer_data[arg0_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                in_shape);
                        };
                    }
                    else if (index_element_type == element::i64)
                    {
                        functor = [&,
                                   in_shape,
                                   element_count,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<float, int64_t>(
                                static_cast<int64_t*>(ctx->buffer_data[arg0_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[arg1_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                in_shape);
                        };
                    }
                    else
                    {
                        throw ngraph_error(
                            "Unsupported index type in CPU Builder for EmbeddingLookup");
                    }
                }
                else if (element_type == element::f64)
                {
                    if (index_element_type == element::f32)
                    {
                        functor = [&,
                                   in_shape,
                                   element_count,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<double, float>(
                                static_cast<float*>(ctx->buffer_data[arg0_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[arg1_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                in_shape);
                        };
                    }
                    else if (index_element_type == element::i32)
                    {
                        functor = [&,
                                   in_shape,
                                   element_count,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<double, int>(
                                static_cast<int*>(ctx->buffer_data[arg0_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[arg1_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                in_shape);
                        };
                    }
                    else if (index_element_type == element::i64)
                    {
                        functor = [&,
                                   in_shape,
                                   element_count,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<double, int64_t>(
                                static_cast<int64_t*>(ctx->buffer_data[arg0_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[arg1_buffer_index]),
                                static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                in_shape);
                        };
                    }
                    else
                    {
                        throw ngraph_error(
                            "Unsupported index type in CPU Builder for EmbeddingLookup");
                    }
                }
                else if (element_type == element::i32)
                {
                    if (index_element_type == element::f32)
                    {
                        functor = [&,
                                   in_shape,
                                   element_count,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<int, float>(
                                static_cast<float*>(ctx->buffer_data[arg0_buffer_index]),
                                static_cast<int*>(ctx->buffer_data[arg1_buffer_index]),
                                static_cast<int*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                in_shape);
                        };
                    }
                    else if (index_element_type == element::i32)
                    {
                        functor = [&,
                                   in_shape,
                                   element_count,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<int, int>(
                                static_cast<int*>(ctx->buffer_data[arg0_buffer_index]),
                                static_cast<int*>(ctx->buffer_data[arg1_buffer_index]),
                                static_cast<int*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                in_shape);
                        };
                    }
                    else if (index_element_type == element::i64)
                    {
                        functor = [&,
                                   in_shape,
                                   element_count,
                                   arg0_buffer_index,
                                   arg1_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<int, int64_t>(
                                static_cast<int64_t*>(ctx->buffer_data[arg0_buffer_index]),
                                static_cast<int*>(ctx->buffer_data[arg1_buffer_index]),
                                static_cast<int*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                in_shape);
                        };
                    }
                    else
                    {
                        throw ngraph_error(
                            "Unsupported index type in CPU Builder for EmbeddingLookup");
                    }
                }
                else
                {
                    throw ngraph_error("Unsupported type in CPU Builder for EmbeddingLookup");
                }

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(EmbeddingLookup);
#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
            void register_builders_embedding_lookup_cpp() {}
#endif
        }
    }
}
