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

#include "ngraph/op/argmin.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/argmin.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::ArgMin)
            {
                auto& functors = external_function->get_functors();

                const ngraph::op::ArgMin* argmin = static_cast<const ngraph::op::ArgMin*>(node);
                CPUKernelFunctor functor;

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                if (out[0].get_element_type() != element::i64 &&
                    out[0].get_element_type() != element::i32)
                {
                    throw ngraph_error("Unsupported index element type");
                }
                bool is_int64 = out[0].get_element_type() == element::i64;
                auto axis = argmin->get_reduction_axis();
                auto in_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto element_type = args[0].get_element_type();
                if (element_type == element::f32)
                {
                    if (is_int64)
                    {
                        std::function<decltype(runtime::cpu::kernel::argmin<float, int64_t, 1>)>
                            kernel;

                        SELECT_RANK2(
                            kernel, float, int64_t, in_shape.size(), runtime::cpu::kernel::argmin);

                        functor = [&, kernel, in_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            kernel(arg_tensor, out_tensor, in_shape, out_shape, axis, ectx->arena);
                        };
                    }
                    else
                    {
                        std::function<decltype(runtime::cpu::kernel::argmin<float, int, 1>)> kernel;

                        SELECT_RANK2(
                            kernel, float, int, in_shape.size(), runtime::cpu::kernel::argmin);

                        functor = [&, kernel, in_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            kernel(arg_tensor, out_tensor, in_shape, out_shape, axis, ectx->arena);
                        };
                    }
                }
                else if (element_type == element::f64)
                {
                    if (is_int64)
                    {
                        std::function<decltype(runtime::cpu::kernel::argmin<double, int64_t, 1>)>
                            kernel;

                        SELECT_RANK2(
                            kernel, double, int64_t, in_shape.size(), runtime::cpu::kernel::argmin);

                        functor = [&, kernel, in_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            kernel(arg_tensor, out_tensor, in_shape, out_shape, axis, ectx->arena);
                        };
                    }
                    else
                    {
                        std::function<decltype(runtime::cpu::kernel::argmin<double, int, 1>)>
                            kernel;

                        SELECT_RANK2(
                            kernel, double, int, in_shape.size(), runtime::cpu::kernel::argmin);

                        functor = [&, kernel, in_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            kernel(arg_tensor, out_tensor, in_shape, out_shape, axis, ectx->arena);
                        };
                    }
                }
                else if (element_type == element::i32)
                {
                    if (is_int64)
                    {
                        std::function<decltype(runtime::cpu::kernel::argmin<int, int64_t, 1>)>
                            kernel;

                        SELECT_RANK2(
                            kernel, int, int64_t, in_shape.size(), runtime::cpu::kernel::argmin);

                        functor = [&, kernel, in_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            kernel(arg_tensor, out_tensor, in_shape, out_shape, axis, ectx->arena);
                        };
                    }
                    else
                    {
                        std::function<decltype(runtime::cpu::kernel::argmin<int, int, 1>)> kernel;

                        SELECT_RANK2(
                            kernel, int, int, in_shape.size(), runtime::cpu::kernel::argmin);

                        functor = [&, kernel, in_shape, out_shape, axis](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            kernel(arg_tensor, out_tensor, in_shape, out_shape, axis, ectx->arena);
                        };
                    }
                }
                else
                {
                    throw ngraph_error("Unsupported type in CPU Builder for ArgMin");
                }

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(ArgMin);
        }
    }
}
