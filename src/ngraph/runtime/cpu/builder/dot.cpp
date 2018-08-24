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
                    auto functor = [](CPURuntimeContext* ctx) {};
                    functors.emplace_back(functor);
                    return;
                }

                if (!shape_size(arg0_shape) || !shape_size(arg1_shape))
                {
                    auto size = shape_size(result_shape) * out[0].get_element_type().size();
                    auto functor = [&, size](CPURuntimeContext* ctx) {
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

                    auto functor = [&, kernel, element_count](CPURuntimeContext* ctx) {
                        kernel(first_tensor, second_tensor, out_tensor, element_count);
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

                    auto functor =
                        [&, kernel, arg0_shape, arg1_shape, result_shape](CPURuntimeContext* ctx) {
                            kernel(arg0_tensor,
                                   arg1_tensor,
                                   out_tensor,
                                   arg0_shape,
                                   arg1_shape,
                                   result_shape);
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

                    auto functor =
                        [&, kernel, arg0_shape, arg1_shape, result_shape](CPURuntimeContext* ctx) {
                            kernel(arg0_tensor,
                                   arg1_tensor,
                                   out_tensor,
                                   arg0_shape,
                                   arg1_shape,
                                   result_shape);
                        };
                    functors.emplace_back(functor);
                    return;
                }

                if ((arg0_shape.size() == 3) && (arg1_shape.size() == 3) &&
                    reduction_axes_count == 1)
                {
                    std::function<decltype(runtime::cpu::kernel::dot_3d_3d_1rd<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::dot_3d_3d_1rd);

                    auto functor =
                        [&, kernel, arg0_shape, arg1_shape, result_shape](CPURuntimeContext* ctx) {
                            kernel(arg0_tensor,
                                   arg1_tensor,
                                   out_tensor,
                                   arg0_shape,
                                   arg1_shape,
                                   result_shape);
                        };
                    functors.emplace_back(functor);
                    return;
                }

                if ((arg0_shape.size() == 3) && (arg1_shape.size() == 2) &&
                    reduction_axes_count == 1)
                {
                    if (args[0].get_element_type() == element::f32)
                    {
                        auto shape_a = args[0].get_shape();
                        auto shape_b = args[1].get_shape();

                        const int64_t m = shape_a[1];
                        const int64_t k = shape_a[2];
                        const int64_t n = shape_b[1];

                        // this also works when mat_a is shape (1, m, k)
                        const int64_t offset_a = m * k;
                        // we do not offset mat_b
                        const int64_t offset_b = 0;
                        const int64_t offset_c = m * n;

                        const int64_t group_count = 1;
                        const int64_t group_size = shape_a[0];

                        auto functor =
                            [&, offset_a, offset_b, offset_c, m, n, k, group_size, group_count](
                                CPURuntimeContext* ctx) {
                                cblas::Transpose transpose = cblas::Transpose::None;
                                float alpha = 1.0f;

                                vector<const float*> a;
                                for (size_t i = 0; i < group_size; i++)
                                {
                                    a.emplace_back(static_cast<const float*>(arg0_tensor) +
                                                   i * offset_a);
                                }
                                const float** a_array = a.data();
                                int64_t lda_array = std::max(int64_t(1), k);

                                vector<const float*> b;
                                for (size_t i = 0; i < group_size; i++)
                                {
                                    b.emplace_back(static_cast<const float*>(arg1_tensor) +
                                                   i * offset_b);
                                }
                                const float** b_array = b.data();
                                const int64_t ldb_array = std::max(int64_t(1), n);
                                float beta = 0.0f;

                                vector<float*> c;
                                for (size_t i = 0; i < group_size; i++)
                                {
                                    c.emplace_back(static_cast<float*>(out_tensor) + i * offset_c);
                                }
                                float** c_array = c.data();
                                const int64_t ldc_array = std::max(int64_t(1), n);

                                cblas_sgemm_batch(cblas::Layout::RowMajor,
                                                  &transpose,
                                                  &transpose,
                                                  &m,
                                                  &n,
                                                  &k,
                                                  &alpha,
                                                  a_array,
                                                  &lda_array,
                                                  b_array,
                                                  &ldb_array,
                                                  &beta,
                                                  c_array,
                                                  &ldc_array,
                                                  group_count,
                                                  &group_size);
                            };
                        functors.emplace_back(functor);
                        return;
                    }
                    else
                    {
                        std::function<decltype(runtime::cpu::kernel::dot_3d_2d_1rd<float>)> kernel;

                        SELECT_KERNEL(
                            kernel, out[0].get_element_type(), runtime::cpu::kernel::dot_3d_2d_1rd);

                        auto functor = [&, kernel, arg0_shape, arg1_shape, result_shape](
                            CPURuntimeContext* ctx) {
                            kernel(arg0_tensor,
                                   arg1_tensor,
                                   out_tensor,
                                   arg0_shape,
                                   arg1_shape,
                                   result_shape);
                        };
                        functors.emplace_back(functor);
                        return;
                    }
                }

                std::function<decltype(runtime::cpu::kernel::dot<float>)> kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::dot);

                auto functor =
                    [&, kernel, arg0_shape, arg1_shape, result_shape, reduction_axes_count](
                        CPURuntimeContext* ctx) {
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
