//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#define BUILDER_DECL(op_name)                                                                      \
    build<op_name>(CPU_ExternalFunction * external_function,                                       \
                   const ngraph::Node* node,                                                       \
                   const std::vector<TensorViewWrapper>& args,                                     \
                   const std::vector<TensorViewWrapper>& out)

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

#define SELECT_RANK(KV, ET, R, K)                                                                  \
    if (R == 1)                                                                                    \
        KV = K<ET, 1>;                                                                             \
    else if (R == 2)                                                                               \
        KV = K<ET, 2>;                                                                             \
    else if (R == 3)                                                                               \
        KV = K<ET, 3>;                                                                             \
    else if (R == 4)                                                                               \
        KV = K<ET, 4>;                                                                             \
    else if (R == 5)                                                                               \
        KV = K<ET, 5>;                                                                             \
    else if (R == 6)                                                                               \
        KV = K<ET, 6>;                                                                             \
    else if (R == 7)                                                                               \
        KV = K<ET, 7>;                                                                             \
    else                                                                                           \
        throw ngraph_error("Unsupported rank " + std::to_string(R) + " for kernel " #K);

// Per-type and rank kernel macro
#define SELECT_KERNEL_BY_RANK(KV, ET, R, K)                                                        \
    if (ET == element::boolean)                                                                    \
    {                                                                                              \
        SELECT_RANK(KV, char, R, K);                                                               \
    }                                                                                              \
    else if (ET == element::f32)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, float, R, K);                                                              \
    }                                                                                              \
    else if (ET == element::f64)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, double, R, K);                                                             \
    }                                                                                              \
    else if (ET == element::i8)                                                                    \
    {                                                                                              \
        SELECT_RANK(KV, int8_t, R, K);                                                             \
    }                                                                                              \
    else if (ET == element::i16)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, int16_t, R, K);                                                            \
    }                                                                                              \
    else if (ET == element::i32)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, int32_t, R, K);                                                            \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, int64_t, R, K);                                                            \
    }                                                                                              \
    else if (ET == element::u8)                                                                    \
    {                                                                                              \
        SELECT_RANK(KV, uint8_t, R, K);                                                            \
    }                                                                                              \
    else if (ET == element::u16)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, uint16_t, R, K);                                                           \
    }                                                                                              \
    else if (ET == element::u32)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, uint32_t, R, K);                                                           \
    }                                                                                              \
    else if (ET == element::u64)                                                                   \
    {                                                                                              \
        SELECT_RANK(KV, uint64_t, R, K);                                                           \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

// Helper macros for a partial set of element types and ranks
// Useful for keeping compilation time and memory usage reasonable
// when the computed expression is complex

#define PARTIAL_SELECT_RANK(KV, ET, R, K)                                                          \
    if (R == 1)                                                                                    \
        KV = K<ET, 1>;                                                                             \
    else if (R == 2)                                                                               \
        KV = K<ET, 2>;                                                                             \
    else if (R == 3)                                                                               \
        KV = K<ET, 3>;                                                                             \
    else if (R == 4)                                                                               \
        KV = K<ET, 4>;                                                                             \
    else if (R == 5)                                                                               \
        KV = K<ET, 5>;                                                                             \
    else if (R == 6)                                                                               \
        KV = K<ET, 6>;                                                                             \
    else                                                                                           \
        throw ngraph_error("Unsupported rank " + std::to_string(R) + " for kernel " #K);

// Partial per-type and rank kernel macro
#define PARTIAL_SELECT_KERNEL_BY_RANK(KV, ET, R, K)                                                \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        PARTIAL_SELECT_RANK(KV, float, R, K);                                                      \
    }                                                                                              \
    else if (ET == element::f64)                                                                   \
    {                                                                                              \
        PARTIAL_SELECT_RANK(KV, double, R, K);                                                     \
    }                                                                                              \
    else if (ET == element::i8)                                                                    \
    {                                                                                              \
        PARTIAL_SELECT_RANK(KV, int8_t, R, K);                                                     \
    }                                                                                              \
    else if (ET == element::u8)                                                                    \
    {                                                                                              \
        PARTIAL_SELECT_RANK(KV, uint8_t, R, K);                                                    \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported element type " + ET.c_type_string() + " for kernel " #K);  \
    }

#define BUILD_UNARY_ELEMWISE_FUNCTOR(OP)                                                           \
    auto& functors = external_function->get_functors();                                            \
    std::function<void(void*, void*, size_t, int)> kernel;                                         \
                                                                                                   \
    SELECT_KERNEL(kernel, args[0].get_element_type(), OP);                                         \
                                                                                                   \
    auto element_count = out[0].get_size();                                                        \
    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());                    \
    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());                     \
                                                                                                   \
    auto functor = [&, kernel, element_count](CPURuntimeContext* ctx, CPUExecutionContext* ectx) { \
        kernel(arg0_tensor, out0_tensor, element_count, ectx->arena);                              \
    };                                                                                             \
    functors.emplace_back(functor);

#define BUILD_BINARY_ELEMWISE_FUNCTOR(OP)                                                          \
    auto& functors = external_function->get_functors();                                            \
    std::function<void(void*, void*, void*, size_t, int)> kernel;                                  \
                                                                                                   \
    SELECT_KERNEL(kernel, args[0].get_element_type(), OP);                                         \
                                                                                                   \
    auto element_count = out[0].get_size();                                                        \
    auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());                    \
    auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());                    \
    auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());                     \
                                                                                                   \
    auto functor = [&, kernel, element_count](CPURuntimeContext* ctx, CPUExecutionContext* ectx) { \
        kernel(arg0_tensor, arg1_tensor, out0_tensor, element_count, ectx->arena);                 \
    };                                                                                             \
    functors.emplace_back(functor);

#define REGISTER_OP_BUILDER(OP)                                                                    \
    static struct __register_##OP##_builder                                                        \
    {                                                                                              \
        __register_##OP##_builder()                                                                \
        {                                                                                          \
            GetGlobalBuildDispatcher().insert({type_index(typeid(ngraph::op::OP)),                 \
                                               &runtime::cpu::Builder::build<ngraph::op::OP>});    \
        }                                                                                          \
    } __register_##OP##_builder_instance;

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

            class Builder
            {
            public:
                template <typename OP>
                static void build(CPU_ExternalFunction* external_function,
                                  const ngraph::Node* node,
                                  const std::vector<TensorViewWrapper>& args,
                                  const std::vector<TensorViewWrapper>& out)
                {
                    throw unsupported_op("Unimplemented op '" + node->description() +
                                         "' in CPU builder");
                }

                static void nop(CPU_ExternalFunction* external_function,
                                const ngraph::Node* node,
                                const std::vector<TensorViewWrapper>& args,
                                const std::vector<TensorViewWrapper>& out)
                {
                }
            };
        }
    }
}
