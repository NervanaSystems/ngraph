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

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_indices_tensor = external_function->get_tensor_data(out[0].get_name());
                auto& out_values_tensor = external_function->get_tensor_data(out[1].get_name());
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
                        functor = [&, in_shape, out_shape, axis, k, compute_max](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<float, int64_t>(
                                static_cast<float*>(arg_tensor),
                                static_cast<int64_t*>(out_indices_tensor),
                                static_cast<float*>(out_values_tensor),
                                in_shape,
                                out_shape,
                                axis,
                                k,
                                compute_max);
                        };
                    }
                    else
                    {
                        functor = [&, in_shape, out_shape, axis, k, compute_max](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<float, int32_t>(
                                static_cast<float*>(arg_tensor),
                                static_cast<int32_t*>(out_indices_tensor),
                                static_cast<float*>(out_values_tensor),
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
                        functor = [&, in_shape, out_shape, axis, k, compute_max](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<double, int64_t>(
                                static_cast<double*>(arg_tensor),
                                static_cast<int64_t*>(out_indices_tensor),
                                static_cast<double*>(out_values_tensor),
                                in_shape,
                                out_shape,
                                axis,
                                k,
                                compute_max);
                        };
                    }
                    else
                    {
                        functor = [&, in_shape, out_shape, axis, k, compute_max](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::topk<double, int32_t>(
                                static_cast<double*>(arg_tensor),
                                static_cast<int32_t*>(out_indices_tensor),
                                static_cast<double*>(out_values_tensor),
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
                    throw ngraph_error("Unsupported type in CPU Builder for TopK");
                }

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(TopK);
        }
    }
}
