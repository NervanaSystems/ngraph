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
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/atan2.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_equal.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_equal.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/logical_and.hpp"
#include "ngraph/op/logical_not.hpp"
#include "ngraph/op/logical_or.hpp"
#include "ngraph/op/logical_xor.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/round.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/runtime/cpu/cpu_builder_registry.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/kernel/abs.hpp"
#include "ngraph/runtime/cpu/kernel/acos.hpp"
#include "ngraph/runtime/cpu/kernel/add.hpp"
#include "ngraph/runtime/cpu/kernel/and.hpp"
#include "ngraph/runtime/cpu/kernel/asin.hpp"
#include "ngraph/runtime/cpu/kernel/atan.hpp"
#include "ngraph/runtime/cpu/kernel/atan2.hpp"
#include "ngraph/runtime/cpu/kernel/broadcast.hpp"
#include "ngraph/runtime/cpu/kernel/ceil.hpp"
#include "ngraph/runtime/cpu/kernel/cos.hpp"
#include "ngraph/runtime/cpu/kernel/cosh.hpp"
#include "ngraph/runtime/cpu/kernel/cwise_pow.hpp"
#include "ngraph/runtime/cpu/kernel/divide.hpp"
#include "ngraph/runtime/cpu/kernel/equal.hpp"
#include "ngraph/runtime/cpu/kernel/exp.hpp"
#include "ngraph/runtime/cpu/kernel/floor.hpp"
#include "ngraph/runtime/cpu/kernel/greater.hpp"
#include "ngraph/runtime/cpu/kernel/greater_eq.hpp"
#include "ngraph/runtime/cpu/kernel/less.hpp"
#include "ngraph/runtime/cpu/kernel/less_eq.hpp"
#include "ngraph/runtime/cpu/kernel/log.hpp"
#include "ngraph/runtime/cpu/kernel/maximum.hpp"
#include "ngraph/runtime/cpu/kernel/minimum.hpp"
#include "ngraph/runtime/cpu/kernel/multiply.hpp"
#include "ngraph/runtime/cpu/kernel/negative.hpp"
#include "ngraph/runtime/cpu/kernel/not.hpp"
#include "ngraph/runtime/cpu/kernel/not_equal.hpp"
#include "ngraph/runtime/cpu/kernel/or.hpp"
#include "ngraph/runtime/cpu/kernel/relu.hpp"
#include "ngraph/runtime/cpu/kernel/result.hpp"
#include "ngraph/runtime/cpu/kernel/round.hpp"
#include "ngraph/runtime/cpu/kernel/sign.hpp"
#include "ngraph/runtime/cpu/kernel/sin.hpp"
#include "ngraph/runtime/cpu/kernel/sinh.hpp"
#include "ngraph/runtime/cpu/kernel/sqrt.hpp"
#include "ngraph/runtime/cpu/kernel/subtract.hpp"
#include "ngraph/runtime/cpu/kernel/tan.hpp"
#include "ngraph/runtime/cpu/kernel/tanh.hpp"
#include "ngraph/runtime/cpu/kernel/xor.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

#ifdef NGRAPH_CPU_MLIR_ENABLE
#include "contrib/mlir/core/compiler.hpp"
#endif

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::Subtract)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::subtract);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::Multiply)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::multiply);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::Divide)
            {
                auto& functors = external_function->get_functors();
                const ngraph::op::v1::Divide* divop =
                    static_cast<const ngraph::op::v1::Divide*>(node);
                std::function<void(void*, void*, void*, size_t, bool, int)> kernel;
                SELECT_KERNEL(kernel, args[0].get_element_type(), runtime::cpu::kernel::divide)
                auto element_count = out[0].get_size();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                bool pythondiv = divop->is_pythondiv();
                auto functor = [&,
                                kernel,
                                element_count,
                                arg0_buffer_index,
                                arg1_buffer_index,
                                out0_buffer_index,
                                pythondiv](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                    kernel(ctx->buffer_data[arg0_buffer_index],
                           ctx->buffer_data[arg1_buffer_index],
                           ctx->buffer_data[out0_buffer_index],
                           element_count,
                           pythondiv,
                           ectx->arena);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::Equal)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::equal);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::NotEqual)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::not_equal);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::Greater)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::greater);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::GreaterEqual)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::greater_eq);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::Less)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::less);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::LessEqual)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::less_equal);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::LogicalAnd)
            {
                (void)node;
                auto& functors = external_function->get_functors();

                auto element_count = out[0].get_size();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto functor =
                    [&, element_count, arg0_buffer_index, arg1_buffer_index, out0_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        runtime::cpu::kernel::logical_and(ctx->buffer_data[arg0_buffer_index],
                                                          ctx->buffer_data[arg1_buffer_index],
                                                          ctx->buffer_data[out0_buffer_index],
                                                          element_count,
                                                          ectx->arena);
                    };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::LogicalOr)
            {
                (void)node;
                auto& functors = external_function->get_functors();

                auto element_count = out[0].get_size();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto functor =
                    [&, element_count, arg0_buffer_index, arg1_buffer_index, out0_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        runtime::cpu::kernel::logical_or(ctx->buffer_data[arg0_buffer_index],
                                                         ctx->buffer_data[arg1_buffer_index],
                                                         ctx->buffer_data[out0_buffer_index],
                                                         element_count,
                                                         ectx->arena);
                    };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::LogicalXor)
            {
                (void)node;
                auto& functors = external_function->get_functors();

                auto element_count = out[0].get_size();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto functor =
                    [&, element_count, arg0_buffer_index, arg1_buffer_index, out0_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        runtime::cpu::kernel::logical_xor(ctx->buffer_data[arg0_buffer_index],
                                                          ctx->buffer_data[arg1_buffer_index],
                                                          ctx->buffer_data[out0_buffer_index],
                                                          element_count,
                                                          ectx->arena);
                    };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::Maximum)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::maximum);
            }
            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::Minimum)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::minimum);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::Power)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::cwise_pow);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Abs)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::abs);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Acos)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::acos);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Asin)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::asin);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Atan)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::atan);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Atan2)
            {
                BUILD_BINARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::atan2);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Ceiling)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::ceil);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Cos)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::cos);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Cosh)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::cosh);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Floor)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::floor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Round)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::round);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Negative)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::negative);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Sqrt)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::sqrt);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Result)
            {
                if (args[0].get_element_type() == element::bf16)
                {
                    auto& functors = external_function->get_functors();
                    std::function<void(void*, void*, size_t, int)> kernel;

                    kernel = ngraph::runtime::cpu::kernel::result<bfloat16>;

                    auto element_count = out[0].get_size();
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto functor = [&, kernel, element_count, arg0_buffer_index, out0_buffer_index](
                                       CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[out0_buffer_index],
                               element_count,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::result);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Exp)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::exp);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Log)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::log);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v1::LogicalNot)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::logical_not);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Sign)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::sign);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Sin)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::sin);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Sinh)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::sinh);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Tan)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::tan);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Tanh)
            {
                BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::tanh);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::v0::Constant)
            {
                (void)args;
                (void)out;
                auto& functors = external_function->get_functors();

                vector<size_t> dest_indices;
                for (auto& result : external_function->get_function()->get_results())
                {
                    if (result.get() == node)
                    {
                        dest_indices.push_back(external_function->get_buffer_index(
                            result->get_output_tensor(0).get_name()));
                    }
                }
                auto src_index =
                    external_function->get_buffer_index(node->get_output_tensor(0).get_name());
                auto size = node->get_output_tensor(0).size();
                auto functor = [&, dest_indices, src_index, size](CPURuntimeContext* ctx,
                                                                  CPUExecutionContext* /* ectx */) {
                    for (auto p : dest_indices)
                    {
                        memcpy(ctx->buffer_data[p], ctx->buffer_data[src_index], size);
                    }
                };
                functors.emplace_back(functor);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Add)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::add);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Subtract)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::subtract);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Multiply)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::multiply);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Power)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::cwise_pow);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Divide)
            {
                const ngraph::op::v1::Divide* divop =
                    static_cast<const ngraph::op::v1::Divide*>(node);
                std::function<void(void*, void*, void*, size_t, bool, int)> kernel;
                SELECT_KERNEL(kernel, node->get_input_element_type(0), runtime::cpu::kernel::divide)
                auto element_count = shape_size(node->get_output_shape(0));
                bool pythondiv = divop->is_pythondiv();
                auto functor = [&, kernel, element_count, pythondiv](
                                   const std::vector<void*>& inputs, std::vector<void*>& outputs) {
                    kernel(inputs[0], inputs[1], outputs[0], element_count, pythondiv, 0);
                };
                return functor;
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Minimum)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::minimum);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Maximum)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::maximum);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v0::Abs)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::abs);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v0::Negative)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::negative);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v0::Relu)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::relu);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v0::Sqrt)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::checked_sqrt);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v0::Floor)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::floor);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v0::Round)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::round);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v0::Ceiling)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::ceil);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Equal)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::equal);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::NotEqual)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::not_equal);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Greater)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::greater);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::GreaterEqual)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::greater_eq);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::Less)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::less);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::LessEqual)
            {
                BUILD_BINARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::less_equal);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::LogicalAnd)
            {
                auto element_count = shape_size(node->get_output_shape(0));

                auto functor = [&, element_count](const std::vector<void*>& inputs,
                                                  std::vector<void*>& outputs) {
                    runtime::cpu::kernel::logical_and(
                        inputs[0], inputs[1], outputs[0], element_count, 0);
                };
                return functor;
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::LogicalOr)
            {
                auto element_count = shape_size(node->get_output_shape(0));

                auto functor = [&, element_count](const std::vector<void*>& inputs,
                                                  std::vector<void*>& outputs) {
                    runtime::cpu::kernel::logical_or(
                        inputs[0], inputs[1], outputs[0], element_count, 0);
                };
                return functor;
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::LogicalXor)
            {
                auto element_count = shape_size(node->get_output_shape(0));

                auto functor = [&, element_count](const std::vector<void*>& inputs,
                                                  std::vector<void*>& outputs) {
                    runtime::cpu::kernel::logical_xor(
                        inputs[0], inputs[1], outputs[0], element_count, 0);
                };
                return functor;
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v0::Sign)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::sign);
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::v1::LogicalNot)
            {
                BUILD_UNARY_ELEMWISE_CF_FUNCTOR(runtime::cpu::kernel::logical_not);
            }

#define TI(x) type_index(typeid(x))

            BuildOpMap& GetGlobalBuildDispatcher()
            {
                static BuildOpMap build_dispatcher{
                    {TI(ngraph::op::v0::Parameter), &runtime::cpu::Builder::nop},
                    {TI(ngraph::op::v0::CompiledKernel),
                     &runtime::cpu::Builder::build<ngraph::op::v0::CompiledKernel>}};

                return build_dispatcher;
            }

            BuildNodeExecutorMap& GetGlobalCFDispatcherCPU()
            {
                static BuildNodeExecutorMap build_cf_dispatcher_cpu{};
                return build_cf_dispatcher_cpu;
            }

            void register_cpu_builders()
            {
                REGISTER_OP_BUILDER(ngraph::op::v0::Constant);
                REGISTER_OP_BUILDER(ngraph::op::v0::Result);
                REGISTER_OP_BUILDER(ngraph::op::v1::Subtract);
                REGISTER_OP_BUILDER(ngraph::op::v1::Multiply);
                REGISTER_OP_BUILDER(ngraph::op::v1::Divide);
                REGISTER_OP_BUILDER(ngraph::op::v1::Power);
                REGISTER_OP_BUILDER(ngraph::op::v0::Abs);
                REGISTER_OP_BUILDER(ngraph::op::v0::Acos);
                REGISTER_OP_BUILDER(ngraph::op::v0::Asin);
                REGISTER_OP_BUILDER(ngraph::op::v0::Atan);
                REGISTER_OP_BUILDER(ngraph::op::v0::Atan2);
                REGISTER_OP_BUILDER(ngraph::op::v0::Ceiling);
                REGISTER_OP_BUILDER(ngraph::op::v0::Cos);
                REGISTER_OP_BUILDER(ngraph::op::v0::Cosh);
                REGISTER_OP_BUILDER(ngraph::op::v0::Floor);
                REGISTER_OP_BUILDER(ngraph::op::v0::Negative);
                REGISTER_OP_BUILDER(ngraph::op::v0::Exp);
                REGISTER_OP_BUILDER(ngraph::op::v0::Log);
                REGISTER_OP_BUILDER(ngraph::op::v0::Round);
                REGISTER_OP_BUILDER(ngraph::op::v0::Sqrt);
                REGISTER_OP_BUILDER(ngraph::op::v0::Sign);
                REGISTER_OP_BUILDER(ngraph::op::v0::Sin);
                REGISTER_OP_BUILDER(ngraph::op::v0::Sinh);
                REGISTER_OP_BUILDER(ngraph::op::v0::Tan);
                REGISTER_OP_BUILDER(ngraph::op::v0::Tanh);

                REGISTER_OP_BUILDER(ngraph::op::v1::LogicalNot);
                REGISTER_OP_BUILDER(ngraph::op::v1::Equal);
                REGISTER_OP_BUILDER(ngraph::op::v1::NotEqual);
                REGISTER_OP_BUILDER(ngraph::op::v1::Greater);
                REGISTER_OP_BUILDER(ngraph::op::v1::GreaterEqual);
                REGISTER_OP_BUILDER(ngraph::op::v1::Less);
                REGISTER_OP_BUILDER(ngraph::op::v1::LessEqual);
                REGISTER_OP_BUILDER(ngraph::op::v1::Maximum);
                REGISTER_OP_BUILDER(ngraph::op::v1::Minimum);
                REGISTER_OP_BUILDER(ngraph::op::v1::LogicalAnd);
                REGISTER_OP_BUILDER(ngraph::op::v1::LogicalOr);
                REGISTER_OP_BUILDER(ngraph::op::v1::LogicalXor);

                REGISTER_CF_BUILDER(ngraph::op::v1::Add);
                REGISTER_CF_BUILDER(ngraph::op::v1::Subtract);
                REGISTER_CF_BUILDER(ngraph::op::v1::Multiply);
                REGISTER_CF_BUILDER(ngraph::op::v1::Divide);
                REGISTER_CF_BUILDER(ngraph::op::v1::Minimum);
                REGISTER_CF_BUILDER(ngraph::op::v1::Maximum);
                REGISTER_CF_BUILDER(ngraph::op::v0::Abs);
                REGISTER_CF_BUILDER(ngraph::op::v0::Negative);
                REGISTER_CF_BUILDER(ngraph::op::v0::Relu);
                REGISTER_CF_BUILDER(ngraph::op::v0::Sqrt);
                REGISTER_CF_BUILDER(ngraph::op::v0::Floor);
                REGISTER_CF_BUILDER(ngraph::op::v0::Ceiling);
                REGISTER_CF_BUILDER(ngraph::op::v1::Equal);
                REGISTER_CF_BUILDER(ngraph::op::v1::NotEqual);
                REGISTER_CF_BUILDER(ngraph::op::v1::Greater);
                REGISTER_CF_BUILDER(ngraph::op::v1::GreaterEqual);
                REGISTER_CF_BUILDER(ngraph::op::v1::Less);
                REGISTER_CF_BUILDER(ngraph::op::v1::LessEqual);
                REGISTER_CF_BUILDER(ngraph::op::v1::LogicalAnd);
                REGISTER_CF_BUILDER(ngraph::op::v1::LogicalOr);
                REGISTER_CF_BUILDER(ngraph::op::v1::LogicalXor);
                REGISTER_CF_BUILDER(ngraph::op::v0::Round);
                REGISTER_CF_BUILDER(ngraph::op::v0::Sign);
                REGISTER_CF_BUILDER(ngraph::op::v1::LogicalNot);
                REGISTER_CF_BUILDER(ngraph::op::v1::Power);
            }
        }
    }
}
