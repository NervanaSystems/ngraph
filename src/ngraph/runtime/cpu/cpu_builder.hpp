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

#pragma once

#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/kernel_selectors.hpp"

#define BUILDER_DECL(op_name)                                                                      \
    build<op_name>(CPU_ExternalFunction * external_function,                                       \
                   const ngraph::Node* node,                                                       \
                   const std::vector<TensorViewWrapper>& args,                                     \
                   const std::vector<TensorViewWrapper>& out)

#define BUILD_UNARY_ELEMWISE_FUNCTOR(OP)                                                           \
    (void)node;                                                                                    \
    auto& functors = external_function->get_functors();                                            \
    std::function<void(void*, void*, size_t, int)> kernel;                                         \
                                                                                                   \
    SELECT_KERNEL(kernel, args[0].get_element_type(), OP);                                         \
                                                                                                   \
    auto element_count = out[0].get_size();                                                        \
    auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());              \
    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());               \
                                                                                                   \
    auto functor = [&, kernel, element_count, arg0_buffer_index, out0_buffer_index](               \
        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {                                       \
        kernel(ctx->buffer_data[arg0_buffer_index],                                                \
               ctx->buffer_data[out0_buffer_index],                                                \
               element_count,                                                                      \
               ectx->arena);                                                                       \
    };                                                                                             \
    functors.emplace_back(functor)

#define BUILD_BINARY_ELEMWISE_FUNCTOR(OP)                                                          \
    (void)node;                                                                                    \
    auto& functors = external_function->get_functors();                                            \
    std::function<void(void*, void*, void*, size_t, int)> kernel;                                  \
                                                                                                   \
    SELECT_KERNEL(kernel, args[0].get_element_type(), OP);                                         \
                                                                                                   \
    auto element_count = out[0].get_size();                                                        \
    auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());              \
    auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());              \
    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());               \
                                                                                                   \
    auto functor =                                                                                 \
        [&, kernel, element_count, arg0_buffer_index, arg1_buffer_index, out0_buffer_index](       \
            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {                                   \
            kernel(ctx->buffer_data[arg0_buffer_index],                                            \
                   ctx->buffer_data[arg1_buffer_index],                                            \
                   ctx->buffer_data[out0_buffer_index],                                            \
                   element_count,                                                                  \
                   ectx->arena);                                                                   \
        };                                                                                         \
    functors.emplace_back(functor)

#define BUILD_UNARY_ELEMWISE_CF_FUNCTOR(OP)                                                        \
    std::function<void(void*, void*, size_t, int)> kernel;                                         \
                                                                                                   \
    SELECT_KERNEL(kernel, node->get_input_element_type(0), OP);                                    \
                                                                                                   \
    auto element_count = shape_size(node->get_shape());                                            \
                                                                                                   \
    auto functor = [&, kernel, element_count](const std::vector<void*>& inputs,                    \
                                              std::vector<void*>& outputs) {                       \
        kernel(inputs[0], outputs[0], element_count, 0);                                           \
    };                                                                                             \
    return functor

#define BUILD_BINARY_ELEMWISE_CF_FUNCTOR(OP)                                                       \
    std::function<void(void*, void*, void*, size_t, int)> kernel;                                  \
                                                                                                   \
    SELECT_KERNEL(kernel, node->get_input_element_type(0), OP);                                    \
                                                                                                   \
    auto element_count = shape_size(node->get_shape());                                            \
                                                                                                   \
    auto functor = [&, kernel, element_count](const std::vector<void*>& inputs,                    \
                                              std::vector<void*>& outputs) {                       \
        kernel(inputs[0], inputs[1], outputs[0], element_count, 0);                                \
    };                                                                                             \
    return functor

#define REGISTER_OP_BUILDER(OP)                                                                    \
    GetGlobalBuildDispatcher().insert(                                                             \
        {type_index(typeid(ngraph::op::OP)), &runtime::cpu::Builder::build<ngraph::op::OP>})

#define REGISTER_CPU_OP_BUILDER(OP)                                                                \
    GetGlobalBuildDispatcher().insert(                                                             \
        {type_index(typeid(ngraph::runtime::cpu::op::OP)),                                         \
         &runtime::cpu::Builder::build<ngraph::runtime::cpu::op::OP>})

#define BUILDER_CF_DECL(op_name) CFbuild<op_name>(const ngraph::Node* node)

#define REGISTER_CF_BUILDER(OP)                                                                    \
    GetGlobalCFDispatcherCPU().insert(                                                             \
        {type_index(typeid(ngraph::op::OP)), &runtime::cpu::Builder::CFbuild<ngraph::op::OP>})

#define REGISTER_CPU_CF_BUILDER(OP)                                                                \
    GetGlobalCFDispatcherCPU().insert(                                                             \
        {type_index(typeid(ngraph::runtime::cpu::op::OP)),                                         \
         &runtime::cpu::Builder::CFbuild<ngraph::runtime::cpu::op::OP>})

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            using BuildOpFunction =
                std::function<void(CPU_ExternalFunction* external_function,
                                   const ngraph::Node*,
                                   const std::vector<TensorViewWrapper>& inputs,
                                   const std::vector<TensorViewWrapper>& outputs)>;

            using BuildOpMap = std::unordered_map<std::type_index, BuildOpFunction>;

            BuildOpMap& GetGlobalBuildDispatcher();

            // build the map to use cpu kernel for node execution
            CPU_BACKEND_API BuildNodeExecutorMap& GetGlobalCFDispatcherCPU();

            class Builder
            {
            public:
                template <typename OP>
                static void build(CPU_ExternalFunction* /* external_function */,
                                  const ngraph::Node* node,
                                  const std::vector<TensorViewWrapper>& /* args */,
                                  const std::vector<TensorViewWrapper>& /* out */)
                {
                    throw unsupported_op("Unimplemented op '" + node->description() +
                                         "' in CPU builder");
                }

                template <typename OP>
                static NodeExecutorTy CFbuild(const ngraph::Node* node)
                {
                    throw unsupported_op("Unimplemented op '" + node->description() +
                                         "' for constant folding in CPU builder");
                }

                static void nop(CPU_ExternalFunction* /* external_function */,
                                const ngraph::Node* /* node */,
                                const std::vector<TensorViewWrapper>& /* args */,
                                const std::vector<TensorViewWrapper>& /* out */)
                {
                }
            };
        }
    }
}
