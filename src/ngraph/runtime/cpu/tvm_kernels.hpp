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

#pragma once

#include <iostream>

#include <dmlc/logging.h>
#include <ngraph/except.hpp>
#include <ngraph/shape.hpp>
#include <topi/broadcast.h>
#include <topi/elemwise.h>
#include <topi/nn.h>
#include <topi/transform.h>
#include <topi/x86/default.h>
#include <tvm/build_module.h>
#include <tvm/operation.h>
#include <tvm/tvm.h>
#include "ngraph/node.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"

#define CHECK_BUILD_TVM_FUNCTOR                                                                    \
    (std::getenv("NGRAPH_USE_TVM") != nullptr && args[0].get_element_type() == element::f32 &&     \
     build_tvm_functor(external_function, node, args, out))

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            bool build_tvm_functor(CPU_ExternalFunction* external_function,
                                   const ngraph::Node* node,
                                   const std::vector<TensorViewWrapper>& args,
                                   const std::vector<TensorViewWrapper>& out);
            class TVMInstance
            {
            public:
                TVMInstance();
                ~TVMInstance();
                size_t add_module(tvm::Module& module)
                {
                    m_modules.push_back(module);
                    return m_modules.size() - 1;
                }
                size_t add_module(tvm::Module&& module)
                {
                    m_modules.push_back(module);
                    return m_modules.size() - 1;
                }
                const tvm::runtime::PackedFunc get_func(tvm::Array<tvm::LoweredFunc>& lowered,
                                                        std::string func = "func")
                {
                    m_modules.push_back(
                        std::move(tvm::build(lowered, m_target, tvm::Target(), m_config)));
                    return m_modules[m_modules.size() - 1]->GetFunction(func, false);
                }
                const tvm::runtime::PackedFunc get_func(const tvm::Array<tvm::Tensor>& G)
                {
                    std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
                    auto schedule = topi::x86::default_schedule(m_target, {G[G.size() - 1]});
                    auto lowered = tvm::lower(schedule, G, "func", binds, m_config);
                    return get_func(lowered);
                }
                tvm::Module& get_module(size_t index) { return m_modules[index]; }
                const tvm::BuildConfig& config() const { return m_config; }
                const tvm::Target& target() const { return m_target; }
                const DLContext& context() const { return m_dl_ctx; }
                DLTensor create_dltensor(const DLDataType& type,
                                         const size_t ndim,
                                         tvm_index_t* shape,
                                         void* data);

            private:
                std::vector<tvm::Module> m_modules;
                tvm::BuildConfig m_config;
                tvm::Target m_target;
                DLContext m_dl_ctx;
            };

            namespace tvm_kernel
            {
                // Unary element wise kernels
                using UnaryElemwiseFunc =
                    std::function<tvm::Tensor(const tvm::Tensor&, std::string, std::string)>;
                using UnaryElemwiseFuncPtr = tvm::Tensor (*)(const tvm::Tensor&,
                                                             std::string,
                                                             std::string);
                template <typename ElementType>
                void unary_elemwise_kernel(const std::unique_ptr<TVMInstance>& tvm_instance,
                                           const tvm::runtime::PackedFunc& func,
                                           void* input,
                                           void* output,
                                           size_t count)
                {
                    throw ngraph_error(
                        "tvm_kernel::unary_elemwise_kernel() instantiated with "
                        "unsupported ElementType");
                }
                template <>
                void unary_elemwise_kernel<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                                  const tvm::runtime::PackedFunc& func,
                                                  void* input,
                                                  void* output,
                                                  size_t count);

                // Unary element wise builders
                template <typename ElementType>
                tvm::PackedFunc
                    unary_elemwise_builder(const std::unique_ptr<TVMInstance>& tvm_instance,
                                           const UnaryElemwiseFunc& topi_func)
                {
                    throw ngraph_error(
                        "tvm_kernel::unary_elemwise_builder() instantiated with "
                        "unsupported ElementType");
                }

                template <>
                tvm::PackedFunc
                    unary_elemwise_builder<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                                  const UnaryElemwiseFunc& topi_func);
                using UnaryElemwiseBuilder = std::function<decltype(unary_elemwise_builder<float>)>;
                using UnaryElemwiseKernel = std::function<decltype(unary_elemwise_kernel<float>)>;

                // Binary element wise kernels
                using BinaryElemwiseFunc = std::function<tvm::Tensor(
                    const tvm::Tensor&, const tvm::Tensor&, std::string, std::string)>;
                using BinaryElemwiseFuncPtr = tvm::Tensor (*)(const tvm::Tensor&,
                                                              const tvm::Tensor&,
                                                              std::string,
                                                              std::string);
                template <typename ElementType>
                void binary_elemwise_kernel(const std::unique_ptr<TVMInstance>& tvm_instance,
                                            const tvm::runtime::PackedFunc& func,
                                            void* input0,
                                            void* input1,
                                            void* output,
                                            size_t count)
                {
                    throw ngraph_error(
                        "tvm_kernel::binary_elemwise_kernel() instantiated with "
                        "unsupported ElementType");
                }
                template <>
                void binary_elemwise_kernel<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                                   const tvm::runtime::PackedFunc& func,
                                                   void* input0,
                                                   void* input1,
                                                   void* output,
                                                   size_t count);

                // Binary element wise builders
                template <typename ElementType>
                tvm::PackedFunc
                    binary_elemwise_builder(const std::unique_ptr<TVMInstance>& tvm_instance,
                                            const BinaryElemwiseFunc& topi_func)
                {
                    throw ngraph_error(
                        "tvm_kernel::binary_elemwise_builder() instantiated with "
                        "unsupported ElementType");
                }

                template <>
                tvm::PackedFunc
                    binary_elemwise_builder<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                                   const BinaryElemwiseFunc& topi_func);
                using BinaryElemwiseBuilder =
                    std::function<decltype(binary_elemwise_builder<float>)>;
                using BinaryElemwiseKernel = std::function<decltype(binary_elemwise_kernel<float>)>;

                // Transpose
                template <typename ElementType>
                tvm::PackedFunc transpose_builder(const std::unique_ptr<TVMInstance>& tvm_instance,
                                                  const size_t in_rank,
                                                  const std::vector<size_t>& in_shape,
                                                  const size_t out_rank,
                                                  const std::vector<size_t>& axes)
                {
                    throw ngraph_error(
                        "tvm_kernel::transpose_builder() instantiated with "
                        "unsupported ElementType");
                }
                template <>
                tvm::PackedFunc
                    transpose_builder<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                             const size_t in_rank,
                                             const std::vector<size_t>& in_shape,
                                             const size_t out_rank,
                                             const std::vector<size_t>& axes);

                template <typename ElementType>
                void transpose_kernel(const std::unique_ptr<TVMInstance>& tvm_instance,
                                      const tvm::PackedFunc& func,
                                      void* input,
                                      void* output,
                                      Shape input_shape,
                                      Shape output_shape)
                {
                    throw ngraph_error(
                        "tvm_kernel::transpose_kernel() instantiated with "
                        "unsupported ElementType");
                }

                template <>
                void transpose_kernel<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                             const tvm::PackedFunc& func,
                                             void* input,
                                             void* output,
                                             ngraph::Shape input_shape,
                                             ngraph::Shape output_shape);
#if 0

                template <unsigned int InRank, unsigned int OutRank>
                tvm::PackedFunc transpose_builder<float, InRank, OutRank>(const std::unique_ptr<TVMInstance>& tvm_instance)
                {
                  tvm::Array<tvm::Expr> in_shape;
                  for (unsigned int i = 0; i < InRank; ++i) {
                      tvm::Var n("in"+std::to_string(i));
                      in_shape.push_back(n);
                  }
                  tvm::Array<tvm::Expr> out_shape;
                  for (unsigned int i = 0; i < InRank; ++i) {
                      tvm::Var n("out"+std::to_string(i));
                      out_shape.push_back(n);
                  }
                  auto A = tvm::placeholder(in_shape, tvm::Float(32), "a");

                  auto R = topi::transpose(A, out_shape);

                  std::unordered_map<tvm::Tensor, tvm::Buffer> binds;

                  auto schedule = topi::x86::default_schedule(tvm_instance->target(), {R});
                  auto lowered = tvm::lower(schedule, {A, R}, "func", binds, tvm_instance->config());
                  auto module =
                      tvm::build(lowered, tvm_instance->target(), tvm::Target(), tvm_instance->config());
                  // store module to keep its lifetime
                  tvm_instance->add_module(module);
                  return module->GetFunction("func", false);
                }
#endif
                using TransposeBuilder = std::function<decltype(transpose_builder<float>)>;
                using TransposeKernel = std::function<decltype(transpose_kernel<float>)>;
            }
        }
    }
}
