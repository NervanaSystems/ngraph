/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstring>

#include "ngraph/op/argmax.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/reference/argmax.hpp"

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
                auto& tensor_data = external_function->get_tensor_data();

                const ngraph::op::ArgMax* argmax = static_cast<const ngraph::op::ArgMax*>(node);
                function<void(CPURuntimeContext*)> functor;

                auto& arg_tensor = tensor_data[args[0].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];
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
                        functor = [&, in_shape, out_shape, axis](CPURuntimeContext* ctx) {
                            ngraph::runtime::reference::argmax<float, int64_t>(
                                static_cast<float*>(arg_tensor),
                                static_cast<int64_t*>(out_tensor),
                                in_shape,
                                out_shape,
                                axis);
                        };
                    }
                    else
                    {
                        functor = [&, in_shape, out_shape, axis](CPURuntimeContext* ctx) {
                            ngraph::runtime::reference::argmax<float, int32_t>(
                                static_cast<float*>(arg_tensor),
                                static_cast<int*>(out_tensor),
                                in_shape,
                                out_shape,
                                axis);
                        };
                    }
                }
                else if (element_type == element::f64)
                {
                    if (is_int64)
                    {
                        functor = [&, in_shape, out_shape, axis](CPURuntimeContext* ctx) {
                            ngraph::runtime::reference::argmax<double, int64_t>(
                                static_cast<double*>(arg_tensor),
                                static_cast<int64_t*>(out_tensor),
                                in_shape,
                                out_shape,
                                axis);
                        };
                    }
                    else
                    {
                        functor = [&, in_shape, out_shape, axis](CPURuntimeContext* ctx) {
                            ngraph::runtime::reference::argmax<double, int32_t>(
                                static_cast<double*>(arg_tensor),
                                static_cast<int*>(out_tensor),
                                in_shape,
                                out_shape,
                                axis);
                        };
                    }
                }
                else
                {
                    throw ngraph_error("Unsupported type in CPU Builder for ArgMax");
                }

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(ArgMax);
        }
    }
}
