/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/kernel/abs.hpp"
#include "ngraph/runtime/cpu/kernel/add.hpp"
#include "ngraph/runtime/cpu/kernel/multiply.hpp"
#include "ngraph/runtime/cpu/kernel/result.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
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

// Per-type kernel macro
#define SELECT_KERNEL(KV, ET, K)                                                                   \
    if (ET == element::boolean)                                                                    \
    {                                                                                              \
        KV = K<char>;                                                                              \
    }                                                                                              \
    else if (ET == element::f32)                                                                   \
    {                                                                                              \
        KV = K<float>;                                                                             \
    }                                                                                              \
    else if (ET == element::f64)                                                                   \
    {                                                                                              \
        KV = K<double>;                                                                            \
    }                                                                                              \
    else if (ET == element::i8)                                                                    \
    {                                                                                              \
        KV = K<int8_t>;                                                                            \
    }                                                                                              \
    else if (ET == element::i16)                                                                   \
    {                                                                                              \
        KV = K<int16_t>;                                                                           \
    }                                                                                              \
    else if (ET == element::i32)                                                                   \
    {                                                                                              \
        KV = K<int32_t>;                                                                           \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        KV = K<int64_t>;                                                                           \
    }                                                                                              \
    else if (ET == element::u8)                                                                    \
    {                                                                                              \
        KV = K<uint8_t>;                                                                           \
    }                                                                                              \
    else if (ET == element::u16)                                                                   \
    {                                                                                              \
        KV = K<uint16_t>;                                                                          \
    }                                                                                              \
    else if (ET == element::u32)                                                                   \
    {                                                                                              \
        KV = K<uint32_t>;                                                                          \
    }                                                                                              \
    else if (ET == element::u64)                                                                   \
    {                                                                                              \
        KV = K<uint64_t>;                                                                          \
    }

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Add)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();
                std::function<void(void*, void*, void*, size_t)> kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::add);

                auto element_count = out[0].get_size();
                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out0_tensor = tensor_data[out[0].get_name()];

                auto functor = [&, kernel, element_count](CPURuntimeContext* ctx) {
                    kernel(arg0_tensor, arg1_tensor, out0_tensor, element_count);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Multiply)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();
                std::function<void(void*, void*, void*, size_t)> kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::multiply);

                auto element_count = out[0].get_size();
                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out0_tensor = tensor_data[out[0].get_name()];

                auto functor = [&, kernel, element_count](CPURuntimeContext* ctx) {
                    kernel(arg0_tensor, arg1_tensor, out0_tensor, element_count);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Abs)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();
                std::function<void(void*, void*, size_t)> kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::abs);

                auto element_count = out[0].get_size();
                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& out0_tensor = tensor_data[out[0].get_name()];

                auto functor = [&, kernel, element_count](CPURuntimeContext* ctx) {
                    kernel(arg0_tensor, out0_tensor, element_count);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Result)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();
                std::function<void(void*, void*, size_t)> kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::result);

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& out0_tensor = tensor_data[out[0].get_name()];
                auto size = shape_size(node->get_shape());

                auto functor = [&, kernel, size](CPURuntimeContext* ctx) {
                    kernel(arg0_tensor, out0_tensor, size);
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
                auto size = node->get_output_tensor(0).size();
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
                {TI(ngraph::op::Result), &runtime::cpu::Builder::build<ngraph::op::Result>},
                {TI(ngraph::op::Constant), &runtime::cpu::Builder::build<ngraph::op::Constant>}};
        }
    }
}
