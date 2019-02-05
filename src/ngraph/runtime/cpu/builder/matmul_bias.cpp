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

#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/op/batch_dot.hpp"

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

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

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
                            max<size_t>(1, lda),
                            static_cast<float*>(arg1_tensor),
                            max<size_t>(1, ldb),
                            beta,
                            static_cast<float*>(out0_tensor),
                            max<size_t>(1, arg2_shape[1]));
                    };

                CPUKernelFunctor bias_functor = [](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* ectx) {};

                if (args.size() > 2)
                {
                    auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());

                    auto axes = mm->get_broadcast_axes();
                    if (axes.size() == 1)
                    {
                        if (*(axes.begin()) == 0)
                        {
                            vector<float> ones_row(arg2_shape[0], 1.0f);
                            bias_functor = [&, ones_row, arg2_shape](CPURuntimeContext* ctx,
                                                                     CPUExecutionContext* ectx) {
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
                                                   max<size_t>(1, arg2_shape[1]),
                                                   1.0f,
                                                   static_cast<float*>(out0_tensor),
                                                   max<size_t>(1, arg2_shape[1]));
                            };
                        }
                        else
                        {
                            vector<float> ones_col(arg2_shape[1], 1.0f);
                            bias_functor = [&, ones_col, arg2_shape](CPURuntimeContext* ctx,
                                                                     CPUExecutionContext* ectx) {
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
                                                   max<size_t>(1, arg2_shape[1]),
                                                   1.0f,
                                                   static_cast<float*>(out0_tensor),
                                                   max<size_t>(1, arg2_shape[1]));
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

                        bias_functor = [&, ones_scalar, arg2_shape](CPURuntimeContext* ctx,
                                                                    CPUExecutionContext* ectx) {
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
                                               max<size_t>(1, arg2_shape[1]),
                                               1.0f,
                                               static_cast<float*>(out0_tensor),
                                               max<size_t>(1, arg2_shape[1]));
                        };
                    }
                }

                auto functor = [&, mm_functor, bias_functor](CPURuntimeContext* ctx,
                                                             CPUExecutionContext* ectx) {
                    mm_functor(ctx, ectx);
                    bias_functor(ctx, ectx);
                };
                functors.emplace_back(functor);
            }

            struct CblasGemmOptions
            {
                CblasGemmOptions(void*& da, void*& db, void*& dc)
                    : data_a(da)
                    , data_b(db)
                    , data_c(dc)
                {
                }

                std::vector<cblas::Transpose> transa_array;
                std::vector<cblas::Transpose> transb_array;
                std::vector<int64_t> m_array;
                std::vector<int64_t> n_array;
                std::vector<int64_t> k_array;
                std::vector<int64_t> lda_array;
                std::vector<int64_t> ldb_array;
                std::vector<int64_t> ldc_array;
                std::vector<int64_t> group_sizes;
                std::vector<float> alpha_array;
                std::vector<float> beta_array;
                size_t offset_a;
                size_t offset_b;
                size_t offset_c;
                void*& data_a;
                void*& data_b;
                void*& data_c;
                int64_t group_count;

                void call(CPURuntimeContext* ctx, CPUExecutionContext* ectx)
                {
                    std::vector<float*> a_array(group_sizes[0]);
                    std::vector<float*> b_array(group_sizes[0]);
                    std::vector<float*> c_array(group_sizes[0]);

                    auto populate_array = [](std::vector<float*>& offsets_vector,
                                             void* data,
                                             int64_t size,
                                             size_t offset) {
                        for (size_t i = 0; i < size; ++i)
                        {
                            offsets_vector.at(i) = static_cast<float*>(data) + (i * offset);
                        }
                    };

                    populate_array(a_array, data_a, group_sizes[0], offset_a);
                    populate_array(b_array, data_b, group_sizes[0], offset_b);
                    populate_array(c_array, data_c, group_sizes[0], offset_c);

                    const float** a = const_cast<const float**>(&a_array[0]);
                    const float** b = const_cast<const float**>(&b_array[0]);

                    cblas_sgemm_batch(cblas::Layout::RowMajor,
                                      &transa_array[0],
                                      &transb_array[0],
                                      &m_array[0],
                                      &n_array[0],
                                      &k_array[0],
                                      &alpha_array[0],
                                      a,
                                      &lda_array[0],
                                      b,
                                      &ldb_array[0],
                                      &beta_array[0],
                                      &c_array[0],
                                      &ldc_array[0],
                                      group_count,
                                      &group_sizes[0]);
                }
            };

            static CPUKernelFunctor emitCblasSgemmBatch(const Shape& shape_a,
                                                        const Shape& shape_b,
                                                        const Shape& shape_c,
                                                        bool transpose_a,
                                                        bool transpose_b,
                                                        void*& data_a,
                                                        void*& data_b,
                                                        void*& data_c,
                                                        const float alpha,
                                                        const float beta,
                                                        size_t group_size)
            {
                size_t m = shape_a[1];
                size_t k = shape_a[2];
                size_t n = shape_b[2];
                size_t lda = std::max<size_t>(1, k);
                size_t ldb = std::max<size_t>(1, n);
                cblas::Transpose ctranspose_a = cblas::Transpose::None;
                cblas::Transpose ctranspose_b = cblas::Transpose::None;

                if (transpose_a)
                {
                    ctranspose_a = cblas::Transpose::Transpose;
                    m = shape_a[2];
                    k = shape_a[1];
                    lda = std::max<size_t>(1, m);
                }
                if (transpose_b)
                {
                    ctranspose_b = cblas::Transpose::Transpose;
                    n = shape_b[1];
                    ldb = std::max<size_t>(1, k);
                }
                size_t ldc = std::max<size_t>(1, n);

                CblasGemmOptions options(data_a, data_b, data_c);

                const size_t offset_a = (shape_a.at(0) > 1) ? m * k : 0;
                const size_t offset_b = (shape_b.at(0) > 1) ? k * n : 0;
                const size_t offset_c = (shape_c.at(0) > 1) ? m * n : 0;

                options.offset_a = offset_a;
                options.offset_b = offset_b;
                options.offset_c = offset_c;

                // if we were to support more groups
                const size_t group_count = 1;
                options.group_count = group_count;

                options.transa_array.push_back(ctranspose_a);
                options.transb_array.push_back(ctranspose_b);

                options.m_array.push_back(m);
                options.n_array.push_back(n);
                options.k_array.push_back(k);

                options.alpha_array.push_back(alpha);
                options.beta_array.push_back(beta);

                options.lda_array.push_back(lda);
                options.ldb_array.push_back(ldb);
                options.ldc_array.push_back(ldc);
                options.group_sizes.push_back(group_size);

                CPUKernelFunctor cblas_func = [options](CPURuntimeContext* ctx,
                                                        CPUExecutionContext* ectx) mutable {
                    options.call(ctx, ectx);
                };
                return cblas_func;
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::BatchDot)
            {
                auto& functors = external_function->get_functors();

                auto& mat_a = external_function->get_tensor_data(args[0].get_name());
                auto& mat_b = external_function->get_tensor_data(args[1].get_name());
                auto& mat_c = external_function->get_tensor_data(out[0].get_name());

                const auto* cg = static_cast<const ngraph::op::BatchDot*>(node);

                const auto& shape_a = cg->get_a_shape();
                const auto& shape_b = cg->get_b_shape();
                const auto& shape_c = out[0].get_shape();

                const size_t group_size = shape_a.at(0);
                auto func = emitCblasSgemmBatch(shape_a,
                                                shape_b,
                                                shape_c,
                                                cg->get_is_a_transposed(),
                                                cg->get_is_b_transposed(),
                                                mat_a,
                                                mat_b,
                                                mat_c,
                                                1.f,
                                                0.f,
                                                group_size);

                functors.emplace_back(func);
            }

            REGISTER_OP_BUILDER(MatmulBias);
            REGISTER_OP_BUILDER(BatchDot);
        }
    }
}
