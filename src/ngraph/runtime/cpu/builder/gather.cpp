//*****************************************************************************
// Copyright 2019 Intel Corporation
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
#include "ngraph/runtime/reference/gather.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Gather)
            {
                auto& functors = external_function->get_functors();
                const ngraph::op::Gather* gather = static_cast<const ngraph::op::Gather*>(node);
                CPUKernelFunctor functor;

                auto& params_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& indices_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                if (args[1].get_element_type() != element::i64 &&
                    args[1].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }
                bool is_int64 = args[1].get_element_type() == element::i64;
                auto axis = gather->get_axis();
                auto params_shape = args[0].get_shape();
                auto indices_shape = args[1].get_shape();
                auto out_shape = out[0].get_shape();
                auto element_type = args[0].get_element_type();
                if (element_type == element::f32)
                {
                    if (is_int64)
                    {
                        functor = [&, params_shape, indices_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::gather<float, int64_t>(
                                static_cast<float*>(params_tensor),
                                static_cast<int64_t*>(indices_tensor),
                                static_cast<float*>(out_tensor),
                                params_shape,
                                indices_shape,
                                out_shape,
                                axis);
                        };
                    }
                    else
                    {
                        functor = [&, params_shape, indices_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::gather<float, int32_t>(
                                static_cast<float*>(params_tensor),
                                static_cast<int32_t*>(indices_tensor),
                                static_cast<float*>(out_tensor),
                                params_shape,
                                indices_shape,
                                out_shape,
                                axis);
                        };
                    }
                }
                else if (element_type == element::f64)
                {
                    if (is_int64)
                    {
                        functor = [&, params_shape, indices_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::gather<double, int64_t>(
                                static_cast<double*>(params_tensor),
                                static_cast<int64_t*>(indices_tensor),
                                static_cast<double*>(out_tensor),
                                params_shape,
                                indices_shape,
                                out_shape,
                                axis);
                        };
                    }
                    else
                    {
                        functor = [&, params_shape, indices_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            ngraph::runtime::reference::gather<double, int32_t>(
                                static_cast<double*>(params_tensor),
                                static_cast<int32_t*>(indices_tensor),
                                static_cast<double*>(out_tensor),
                                params_shape,
                                indices_shape,
                                out_shape,
                                axis);
                        };
                    }
                }
                else
                {
                    throw ngraph_error("Unsupported type in CPU Builder for Gather");
                }

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Gather);
        }
    }
}
