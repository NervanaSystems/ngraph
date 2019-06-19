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

#define SELECT_KERNEL_3ARGS(KV, ET, K)                                                             \
    if (ET == element::boolean)                                                                    \
    {                                                                                              \
        KV = K<char, char, char>;                                                                  \
    }                                                                                              \
    else if (ET == element::f32)                                                                   \
    {                                                                                              \
        KV = K<float, float, float>;                                                               \
    }                                                                                              \
    else if (ET == element::f64)                                                                   \
    {                                                                                              \
        KV = K<double, double, double>;                                                            \
    }                                                                                              \
    else if (ET == element::i8)                                                                    \
    {                                                                                              \
        KV = K<int8_t, int8_t, int8_t>;                                                            \
    }                                                                                              \
    else if (ET == element::i16)                                                                   \
    {                                                                                              \
        KV = K<int16_t, int16_t, int16_t>;                                                         \
    }                                                                                              \
    else if (ET == element::i32)                                                                   \
    {                                                                                              \
        KV = K<int32_t, int32_t, int32_t>;                                                         \
    }                                                                                              \
    else if (ET == element::i64)                                                                   \
    {                                                                                              \
        KV = K<int64_t, int64_t, int64_t>;                                                         \
    }                                                                                              \
    else if (ET == element::u8)                                                                    \
    {                                                                                              \
        KV = K<uint8_t, uint8_t, uint8_t>;                                                         \
    }                                                                                              \
    else if (ET == element::u16)                                                                   \
    {                                                                                              \
        KV = K<uint16_t, uint16_t, uint16_t>;                                                      \
    }                                                                                              \
    else if (ET == element::u32)                                                                   \
    {                                                                                              \
        KV = K<uint32_t, uint32_t, uint32_t>;                                                      \
    }                                                                                              \
    else if (ET == element::u64)                                                                   \
    {                                                                                              \
        KV = K<uint64_t, uint64_t, uint64_t>;                                                      \
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

#define SELECT_RANK2(KV, IT, OT, R, K)                                                             \
    switch (R)                                                                                     \
    {                                                                                              \
    case 1: KV = K<IT, OT, 1>; break;                                                              \
    case 2: KV = K<IT, OT, 2>; break;                                                              \
    case 3: KV = K<IT, OT, 3>; break;                                                              \
    case 4: KV = K<IT, OT, 4>; break;                                                              \
    case 5: KV = K<IT, OT, 5>; break;                                                              \
    case 6: KV = K<IT, OT, 6>; break;                                                              \
    case 7: KV = K<IT, OT, 7>; break;                                                              \
    default: throw ngraph_error("Unsupported rank " + std::to_string(R) + " for kernel " #K);      \
    }

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

#define SELECT_RANK1(KV, ET, R1, R2, K)                                                            \
    if (R1 == 1)                                                                                   \
        KV = K<ET, 1, R2>;                                                                         \
    else if (R1 == 2)                                                                              \
        KV = K<ET, 2, R2>;                                                                         \
    else if (R1 == 3)                                                                              \
        KV = K<ET, 3, R2>;                                                                         \
    else if (R1 == 4)                                                                              \
        KV = K<ET, 4, R2>;                                                                         \
    else if (R1 == 5)                                                                              \
        KV = K<ET, 5, R2>;                                                                         \
    else if (R1 == 6)                                                                              \
        KV = K<ET, 6, R2>;                                                                         \
    else if (R1 == 7)                                                                              \
        KV = K<ET, 7, R2>;                                                                         \
    else                                                                                           \
        throw ngraph_error("Unsupported first rank " + std::to_string(R1) + " for kernel " #K);

#define SELECT_2RANKS(KV, ET, R1, R2, K)                                                           \
    if (R2 == 1)                                                                                   \
    {                                                                                              \
        SELECT_RANK1(KV, ET, R1, 1, K);                                                            \
    }                                                                                              \
    else if (R2 == 2)                                                                              \
    {                                                                                              \
        SELECT_RANK1(KV, ET, R1, 2, K);                                                            \
    }                                                                                              \
    else if (R2 == 3)                                                                              \
    {                                                                                              \
        SELECT_RANK1(KV, ET, R1, 3, K);                                                            \
    }                                                                                              \
    else if (R2 == 4)                                                                              \
    {                                                                                              \
        SELECT_RANK1(KV, ET, R1, 4, K);                                                            \
    }                                                                                              \
    else if (R2 == 5)                                                                              \
    {                                                                                              \
        SELECT_RANK1(KV, ET, R1, 5, K);                                                            \
    }                                                                                              \
    else if (R2 == 6)                                                                              \
    {                                                                                              \
        SELECT_RANK1(KV, ET, R1, 6, K);                                                            \
    }                                                                                              \
    else if (R2 == 7)                                                                              \
    {                                                                                              \
        SELECT_RANK1(KV, ET, R1, 7, K);                                                            \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        throw ngraph_error("Unsupported second rank " + std::to_string(R2) + " for kernel " #K);   \
    }

// Per-type and ranks kernel macro
#define SELECT_KERNEL_BY_2RANKS(KV, ET, R1, R2, K)                                                 \
    if (ET == element::f32)                                                                        \
    {                                                                                              \
        SELECT_2RANKS(KV, float, R1, R2, K);                                                       \
    }                                                                                              \
    else if (ET == element::f64)                                                                   \
    {                                                                                              \
        SELECT_2RANKS(KV, double, R1, R2, K);                                                      \
    }                                                                                              \
    else if (ET == element::u8)                                                                    \
    {                                                                                              \
        SELECT_2RANKS(KV, uint8_t, R1, R2, K);                                                     \
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
    functors.emplace_back(functor);

#define BUILD_BINARY_ELEMWISE_FUNCTOR(OP)                                                          \
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
    functors.emplace_back(functor);

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
    return functor;

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
    return functor;

#define REGISTER_OP_BUILDER(OP)                                                                    \
    static struct __register_##OP##_builder                                                        \
    {                                                                                              \
        __register_##OP##_builder()                                                                \
        {                                                                                          \
            GetGlobalBuildDispatcher().insert({type_index(typeid(ngraph::op::OP)),                 \
                                               &runtime::cpu::Builder::build<ngraph::op::OP>});    \
        }                                                                                          \
    } __register_##OP##_builder_instance;

#define REGISTER_CPU_OP_BUILDER(OP)                                                                \
    static struct __register_##OP##_builder                                                        \
    {                                                                                              \
        __register_##OP##_builder()                                                                \
        {                                                                                          \
            GetGlobalBuildDispatcher().insert(                                                     \
                {type_index(typeid(ngraph::runtime::cpu::op::OP)),                                 \
                 &runtime::cpu::Builder::build<ngraph::runtime::cpu::op::OP>});                    \
        }                                                                                          \
    } __register_##OP##_builder_instance;

#define BUILDER_CF_DECL(op_name) CFbuild<op_name>(const ngraph::Node* node)

#define REGISTER_CF_BUILDER(OP)                                                                    \
    static struct __register_##OP##_cf_builder                                                     \
    {                                                                                              \
        __register_##OP##_cf_builder()                                                             \
        {                                                                                          \
            GetGlobalCFDispatcherCPU().insert({type_index(typeid(ngraph::op::OP)),                 \
                                               &runtime::cpu::Builder::CFbuild<ngraph::op::OP>});  \
        }                                                                                          \
    } __register_##OP##_cf_builder_instance;

#define REGISTER_CPU_CF_BUILDER(OP)                                                                \
    static struct __register_##OP##_cf_builder                                                     \
    {                                                                                              \
        __register_##OP##_cf_builder()                                                             \
        {                                                                                          \
            GetGlobalCFDispatcherCPU().insert(                                                     \
                {type_index(typeid(ngraph::runtime::cpu::op::OP)),                                 \
                 &runtime::cpu::Builder::CFbuild<ngraph::runtime::cpu::op::OP>});                  \
        }                                                                                          \
    } __register_##OP##_cf_builder_instance;

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
                static void build(CPU_ExternalFunction* external_function,
                                  const ngraph::Node* node,
                                  const std::vector<TensorViewWrapper>& args,
                                  const std::vector<TensorViewWrapper>& out)
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
