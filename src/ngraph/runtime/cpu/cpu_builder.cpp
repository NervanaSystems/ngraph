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

#include "ngraph/runtime/cpu/cpu_builder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/function_call.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/remainder.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/kernel/abs.hpp"
#include "ngraph/runtime/cpu/kernel/add.hpp"
#include "ngraph/runtime/cpu/kernel/broadcast.hpp"
#include "ngraph/runtime/cpu/kernel/ceil.hpp"
#include "ngraph/runtime/cpu/kernel/multiply.hpp"
#include "ngraph/runtime/cpu/kernel/relu.hpp"
#include "ngraph/runtime/cpu/kernel/result.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include <mpi.h>
#include "ngraph/op/allreduce.hpp"
#endif

using namespace std;
using namespace ngraph;

#define BUILD_UNARY_ELEMWISE_FUNCTOR(OP)                                                           \
    auto& functors = external_function->get_functors();                                            \
    auto& tensor_data = external_function->get_tensor_data();                                      \
    std::function<void(void*, void*, size_t)> kernel;                                              \
                                                                                                   \
    SELECT_KERNEL(kernel, out[0].get_element_type(), OP);                                          \
                                                                                                   \
    auto element_count = out[0].get_size();                                                        \
    auto& arg0_tensor = tensor_data[args[0].get_name()];                                           \
    auto& out0_tensor = tensor_data[out[0].get_name()];                                            \
                                                                                                   \
    auto functor = [&, kernel, element_count](CPURuntimeContext* ctx) {                            \
        kernel(arg0_tensor, out0_tensor, element_count);                                           \
    };                                                                                             \
    functors.emplace_back(functor);

#define BUILD_BINARY_ELEMWISE_FUNCTOR(OP)                                                          \
    auto& functors = external_function->get_functors();                                            \
    auto& tensor_data = external_function->get_tensor_data();                                      \
    std::function<void(void*, void*, void*, size_t)> kernel;                                       \
                                                                                                   \
    SELECT_KERNEL(kernel, out[0].get_element_type(), OP);                                          \
                                                                                                   \
    auto element_count = out[0].get_size();                                                        \
    auto& arg0_tensor = tensor_data[args[0].get_name()];                                           \
    auto& arg1_tensor = tensor_data[args[1].get_name()];                                           \
    auto& out0_tensor = tensor_data[out[0].get_name()];                                            \
                                                                                                   \
    auto functor = [&, kernel, element_count](CPURuntimeContext* ctx) {                            \
        kernel(arg0_tensor, arg1_tensor, out0_tensor, element_count);                              \
    };                                                                                             \
    functors.emplace_back(functor);

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Add)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::add);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Multiply)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::multiply);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Abs)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::abs);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Broadcast)
            {
                std::function<void(void*, void*, const Shape&, const Shape&, const AxisSet&)>
                    kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::broadcast);

                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto arg0_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                auto broadcast = static_cast<const ngraph::op::Broadcast*>(node);
                auto broadcast_axes = broadcast->get_broadcast_axes();

                auto functor =
                    [&, kernel, arg0_shape, result_shape, broadcast_axes](CPURuntimeContext* ctx) {
                        kernel(arg0_tensor, out_tensor, arg0_shape, result_shape, broadcast_axes);
                    };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Ceiling)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::ceil);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Relu)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::relu);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Result)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::result);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::MatmulBias)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out0_tensor = tensor_data[out[0].get_name()];

                const ngraph::op::MatmulBias* mm = static_cast<const ngraph::op::MatmulBias*>(node);

                const auto& arg0_shape = mm->get_arg0_shape();
                const auto& arg1_shape = mm->get_arg1_shape();
                const auto& arg2_shape = node->get_shape();

                auto m = arg0_shape[0];
                auto n = arg1_shape[1];
                auto k = arg0_shape[1];

                bool transpose_A = false, transpose_B = false;
                auto lda = arg0_shape[1];
                auto ldb = arg1_shape[1];

                if (mm->get_is_arg0_transposed())
                {
                    transpose_A = true;
                    m = arg0_shape[1];
                    k = arg0_shape[0];
                }

                if (mm->get_is_arg1_transposed())
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

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Constant)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                vector<void**> dest;
                for (auto& result : external_function->get_function()->get_results())
                {
                    if (result.get() == node)
                    {
                        dest.push_back(&tensor_data[result->get_output_tensor(0).get_name()]);
                    }
                }
                auto& src = tensor_data[node->get_output_tensor(0).get_name()];
                auto size = node->get_output_tensor(0)
                                .get_primary_tensor_view()
                                ->get_tensor_view_layout()
                                ->size();
                auto functor = [&, dest, src, size](CPURuntimeContext* ctx) {
                    for (auto p : dest)
                    {
                        memcpy(*p, src, size);
                    }
                };
                functors.emplace_back(functor);
            }

#define TI(x) type_index(typeid(x))

            const BuildOpMap build_dispatcher{
                {TI(ngraph::op::Add), &runtime::cpu::Builder::build<ngraph::op::Add>},
                {TI(ngraph::op::Multiply), &runtime::cpu::Builder::build<ngraph::op::Multiply>},
                {TI(ngraph::op::Parameter), &runtime::cpu::Builder::nop},
                {TI(ngraph::op::Abs), &runtime::cpu::Builder::build<ngraph::op::Abs>},
                {TI(ngraph::op::AvgPool), &runtime::cpu::Builder::build<ngraph::op::AvgPool>},
                {TI(ngraph::op::Broadcast), &runtime::cpu::Builder::build<ngraph::op::Broadcast>},
                {TI(ngraph::op::Ceiling), &runtime::cpu::Builder::build<ngraph::op::Ceiling>},
                {TI(ngraph::runtime::cpu::op::ConvertLayout),
                 &runtime::cpu::Builder::build<ngraph::runtime::cpu::op::ConvertLayout>},
                {TI(ngraph::op::Convolution),
                 &runtime::cpu::Builder::build<ngraph::op::Convolution>},
                {TI(ngraph::op::ConvolutionBias),
                 &runtime::cpu::Builder::build<ngraph::op::ConvolutionBias>},
                {TI(ngraph::op::ConvolutionBackpropData),
                 &runtime::cpu::Builder::build<ngraph::op::ConvolutionBackpropData>},
                {TI(ngraph::op::ConvolutionBackpropFilters),
                 &runtime::cpu::Builder::build<ngraph::op::ConvolutionBackpropFilters>},
                {TI(ngraph::op::Relu), &runtime::cpu::Builder::build<ngraph::op::Relu>},
                {TI(ngraph::op::Reshape), &runtime::cpu::Builder::build<ngraph::op::Reshape>},
                {TI(ngraph::op::Result), &runtime::cpu::Builder::build<ngraph::op::Result>},
                {TI(ngraph::op::MatmulBias), &runtime::cpu::Builder::build<ngraph::op::MatmulBias>},
                {TI(ngraph::op::Constant), &runtime::cpu::Builder::build<ngraph::op::Constant>}};
        }
    }
}
