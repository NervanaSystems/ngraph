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

#include <algorithm>
#include <cstring>

#include "ngraph/op/dot.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/kernel/dot.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Dot)
            {
                auto dot = static_cast<const ngraph::op::Dot*>(node);

                auto& functors = external_function->get_functors();

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();
                auto result_shape = out[0].get_shape();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto reduction_axes_count = dot->get_reduction_axes_count();

                if (!shape_size(result_shape))
                {
                    auto functor = [](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {};
                    functors.emplace_back(functor);
                    return;
                }

                if (!shape_size(arg0_shape) || !shape_size(arg1_shape))
                {
                    auto size = shape_size(result_shape) * out[0].get_element_type().size();
                    auto functor = [&, size](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        memset(out_tensor, 0, size);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                if (arg0_shape.empty() || arg1_shape.empty())
                {
                    auto first = (arg0_shape.empty() ? args[0] : args[1]);
                    auto second = (arg0_shape.empty() ? args[1] : args[0]);

                    auto& first_tensor = external_function->get_tensor_data(first.get_name());
                    auto& second_tensor = external_function->get_tensor_data(second.get_name());

                    std::function<decltype(runtime::cpu::kernel::dot_scalar<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::dot_scalar);

                    auto element_count = shape_size(second.get_shape());

                    auto functor = [&, kernel, element_count](CPURuntimeContext* ctx,
                                                              CPUExecutionContext* ectx) {
                        kernel(first_tensor, second_tensor, out_tensor, element_count, ectx->arena);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1) &&
                    reduction_axes_count == 1)
                {
                    std::function<decltype(runtime::cpu::kernel::dot_1d_1d_1rd<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::dot_1d_1d_1rd);

                    auto functor = [&, kernel, arg0_shape, arg1_shape, result_shape](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(arg0_tensor,
                               arg1_tensor,
                               out_tensor,
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) &&
                    reduction_axes_count == 1)
                {
                    std::function<decltype(runtime::cpu::kernel::dot_2d_1d_1rd<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::dot_2d_1d_1rd);

                    auto functor = [&, kernel, arg0_shape, arg1_shape, result_shape](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(arg0_tensor,
                               arg1_tensor,
                               out_tensor,
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                if ((arg0_shape.size() == 1) && (arg1_shape.size() == 2) &&
                    reduction_axes_count == 1)
                {
                    std::function<decltype(runtime::cpu::kernel::dot_1d_2d_1rd<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::dot_1d_2d_1rd);

                    auto functor = [&, kernel, arg0_shape, arg1_shape, result_shape](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(arg0_tensor,
                               arg1_tensor,
                               out_tensor,
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                if (out[0].get_element_type() == element::f32 && (arg0_shape.size() == 2) &&
                    (arg1_shape.size() == 2) && reduction_axes_count == 1)
                {
                    auto m = arg0_shape[0];
                    auto n = arg1_shape[1];
                    auto k = arg0_shape[1];
                    bool transpose_A = false, transpose_B = false;
                    auto lda = arg0_shape[1];
                    auto ldb = arg1_shape[1];
                    const float beta = 0.0f;
                    auto functor =
                        [&, transpose_A, transpose_B, m, n, k, lda, ldb, beta, result_shape](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            cblas::cblas_sgemm(
                                cblas::Layout::RowMajor,
                                transpose_A ? cblas::Transpose::Transpose : cblas::Transpose::None,
                                transpose_B ? cblas::Transpose::Transpose : cblas::Transpose::None,
                                m,
                                n,
                                k,
                                1.0f,
                                static_cast<float*>(arg0_tensor),
                                max<size_t>(1UL, lda),
                                static_cast<float*>(arg1_tensor),
                                max<size_t>(1UL, ldb),
                                beta,
                                static_cast<float*>(out_tensor),
                                max<size_t>(1UL, result_shape[1]));
                        };
                    functors.emplace_back(functor);
                    return;
                }

                std::function<decltype(runtime::cpu::kernel::dot_ref<float>)> kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::dot_ref);

                auto functor =
                    [&, kernel, arg0_shape, arg1_shape, result_shape, reduction_axes_count](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(arg0_tensor,
                               arg1_tensor,
                               out_tensor,
                               arg0_shape,
                               arg1_shape,
                               result_shape,
                               reduction_axes_count);
                    };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Dot);
        }
    }
}
