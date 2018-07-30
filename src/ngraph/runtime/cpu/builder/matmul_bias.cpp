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

#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::MatmulBias)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out0_tensor = tensor_data[out[0].get_name()];

                const ngraph::op::MatmulBias* mm = static_cast<const ngraph::op::MatmulBias*>(node);

                const auto& arg0_shape = mm->get_a_shape();
                const auto& arg1_shape = mm->get_b_shape();
                const auto& arg2_shape = node->get_shape();

                auto m = arg0_shape[0];
                auto n = arg1_shape[1];
                auto k = arg0_shape[1];

                bool transpose_A = false, transpose_B = false;
                auto lda = arg0_shape[1];
                auto ldb = arg1_shape[1];

                if (mm->get_is_a_transposed())
                {
                    transpose_A = true;
                    m = arg0_shape[1];
                    k = arg0_shape[0];
                }

                if (mm->get_is_b_transposed())
                {
                    transpose_B = true;
                    n = arg1_shape[0];
                }

                const float beta = 0.0f;

                auto mm_functor =
                    [&, transpose_A, transpose_B, m, n, k, lda, ldb, beta, arg2_shape](
                        CPURuntimeContext* ctx) {
                        cblas::cblas_sgemm(
                            cblas::Layout::RowMajor,
                            transpose_A ? cblas::Transpose::Transpose : cblas::Transpose::None,
                            transpose_B ? cblas::Transpose::Transpose : cblas::Transpose::None,
                            m,
                            n,
                            k,
                            1.0f,
                            static_cast<float*>(arg0_tensor),
                            max(1UL, lda),
                            static_cast<float*>(arg1_tensor),
                            max(1UL, ldb),
                            beta,
                            static_cast<float*>(out0_tensor),
                            max(1UL, arg2_shape[1]));
                    };

                function<void(CPURuntimeContext*)> bias_functor = [](CPURuntimeContext* ctx) {};

                if (args.size() > 2)
                {
                    auto& arg2_tensor = tensor_data[args[2].get_name()];

                    auto axes = mm->get_broadcast_axes();
                    if (axes.size() == 1)
                    {
                        if (*(axes.begin()) == 0)
                        {
                            vector<float> ones_row(arg2_shape[0], 1.0f);
                            bias_functor = [&, ones_row, arg2_shape](CPURuntimeContext* ctx) {
                                cblas::cblas_sgemm(cblas::Layout::RowMajor,
                                                   cblas::Transpose::None,
                                                   cblas::Transpose::None,
                                                   arg2_shape[0],
                                                   arg2_shape[1],
                                                   1,
                                                   1.0f,
                                                   ones_row.data(),
                                                   1UL,
                                                   static_cast<float*>(arg2_tensor),
                                                   max(1UL, arg2_shape[1]),
                                                   1.0f,
                                                   static_cast<float*>(out0_tensor),
                                                   max(1UL, arg2_shape[1]));
                            };
                        }
                        else
                        {
                            vector<float> ones_col(arg2_shape[1], 1.0f);
                            bias_functor = [&, ones_col, arg2_shape](CPURuntimeContext* ctx) {
                                cblas::cblas_sgemm(cblas::Layout::RowMajor,
                                                   cblas::Transpose::None,
                                                   cblas::Transpose::None,
                                                   arg2_shape[0],
                                                   arg2_shape[1],
                                                   1,
                                                   1.0f,
                                                   static_cast<float*>(arg2_tensor),
                                                   1UL,
                                                   ones_col.data(),
                                                   max(1UL, arg2_shape[1]),
                                                   1.0f,
                                                   static_cast<float*>(out0_tensor),
                                                   max(1UL, arg2_shape[1]));
                            };
                        }
                    }
                    else
                    {
                        if (axes.size() != 2)
                        {
                            throw ngraph_error("unexpected broadcast rank");
                        }

                        vector<float> ones_scalar(arg2_shape[0], 1.0f);

                        bias_functor = [&, ones_scalar, arg2_shape](CPURuntimeContext* ctx) {
                            vector<float> bias(arg2_shape[1], *static_cast<float*>(arg2_tensor));
                            cblas::cblas_sgemm(cblas::Layout::RowMajor,
                                               cblas::Transpose::None,
                                               cblas::Transpose::None,
                                               arg2_shape[0],
                                               arg2_shape[1],
                                               1,
                                               1.0f,
                                               ones_scalar.data(),
                                               1UL,
                                               bias.data(),
                                               max(1UL, arg2_shape[1]),
                                               1.0f,
                                               static_cast<float*>(out0_tensor),
                                               max(1UL, arg2_shape[1]));
                        };
                    }
                }

                auto functor = [&, mm_functor, bias_functor](CPURuntimeContext* ctx) {
                    mm_functor(ctx);
                    bias_functor(ctx);
                };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(MatmulBias);
        }
    }
}
