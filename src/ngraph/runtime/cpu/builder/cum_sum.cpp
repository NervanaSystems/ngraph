//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/runtime/cpu/kernel/cum_sum.hpp"
#include "ngraph/op/cum_sum.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::CumSum)
            {
#define FUNCTOR_CUMSUM(T, M)                                                                       \
    do                                                                                             \
    {                                                                                              \
        auto functor = [&,                                                                         \
                        kernel,                                                                    \
                        arg0_buffer_index,                                                         \
                        arg1_buffer_index,                                                         \
                        out0_buffer_index,                                                         \
                        tensor_shape,                                                              \
                        cumsum_op](CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {      \
            runtime::cpu::kernel::reference_cumsum<T, M>(ctx->buffer_data[arg0_buffer_index],      \
                                                         ctx->buffer_data[arg1_buffer_index],      \
                                                         ctx->buffer_data[out0_buffer_index],      \
                                                         tensor_shape,                             \
                                                         cumsum_op->is_exclusive(),                \
                                                         cumsum_op->is_reverse());                 \
        };                                                                                         \
        functors.emplace_back(functor);                                                            \
    } while (0)
                (void)node;

                auto cumsum_op = static_cast<const ngraph::op::CumSum*>(node);
                auto tensor_shape = args[0].get_shape();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto& functors = external_function->get_functors();

                if (args[0].get_element_type() == element::f32 &&
                    args[1].get_element_type() == element::i32)
                {
                    std::function<decltype(runtime::cpu::kernel::reference_cumsum<float, int32_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(float, int32_t);
                }
                else if (args[0].get_element_type() == element::f32 &&
                         args[1].get_element_type() == element::i64)
                {
                    std::function<decltype(runtime::cpu::kernel::reference_cumsum<float, int64_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(float, int64_t);
                }
                else if (args[0].get_element_type() == element::f64 &&
                         args[1].get_element_type() == element::i32)
                {
                    std::function<decltype(runtime::cpu::kernel::reference_cumsum<double, int32_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(double, int32_t);
                }
                else if (args[0].get_element_type() == element::f64 &&
                         args[1].get_element_type() == element::i64)
                {
                    std::function<decltype(runtime::cpu::kernel::reference_cumsum<double, int64_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(double, int64_t);
                }
                else if (args[0].get_element_type() == element::i32 &&
                         args[1].get_element_type() == element::i32)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::reference_cumsum<int32_t, int32_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(int32_t, int32_t);
                }
                else if (args[0].get_element_type() == element::i32 &&
                         args[1].get_element_type() == element::i64)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::reference_cumsum<int32_t, int64_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(int32_t, int64_t);
                }
                else if (args[0].get_element_type() == element::i64 &&
                         args[1].get_element_type() == element::i32)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::reference_cumsum<int64_t, int32_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(int64_t, int32_t);
                }
                else if (args[0].get_element_type() == element::i64 &&
                         args[1].get_element_type() == element::i64)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::reference_cumsum<int64_t, int64_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(int64_t, int64_t);
                }
                else if (args[0].get_element_type() == element::u32 &&
                         args[1].get_element_type() == element::i32)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::reference_cumsum<uint32_t, int32_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(uint32_t, int32_t);
                }
                else if (args[0].get_element_type() == element::u32 &&
                         args[1].get_element_type() == element::i64)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::reference_cumsum<uint32_t, int64_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(uint32_t, int64_t);
                }
                else if (args[0].get_element_type() == element::u64 &&
                         args[1].get_element_type() == element::i32)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::reference_cumsum<uint64_t, int32_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(uint64_t, int32_t);
                }
                else if (args[0].get_element_type() == element::u64 &&
                         args[1].get_element_type() == element::i64)
                {
                    std::function<decltype(
                        runtime::cpu::kernel::reference_cumsum<uint64_t, int64_t>)>
                        kernel;
                    FUNCTOR_CUMSUM(uint64_t, int64_t);
                }
            }

            void register_builders_cumsum_cpp() { REGISTER_OP_BUILDER(CumSum); }
        }
    }
}
