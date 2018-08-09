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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <ngraph/except.hpp>
#include <topi/broadcast.h>
#include <topi/elemwise.h>
#include <topi/nn.h>
#include <tvm/build_module.h>
#include <tvm/tvm.h>

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class TVMInstance
            {
            public:
                TVMInstance();
                ~TVMInstance();
                size_t add_module(const tvm::Module& module)
                {
                    m_modules.push_back(module);
                    return m_modules.size() - 1;
                }
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
                typedef std::function<tvm::Tensor(const tvm::Tensor&, std::string, std::string)>
                    UnaryElemwiseFunc;
                typedef tvm::Tensor (*UnaryElemwiseFuncPtr)(const tvm::Tensor&,
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
                typedef std::function<tvm::Tensor(
                    const tvm::Tensor&, const tvm::Tensor&, std::string, std::string)>
                    BinaryElemwiseFunc;
                typedef tvm::Tensor (*BinaryElemwiseFuncPtr)(const tvm::Tensor&,
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

                // Relu
                using ReluFunc = std::function<decltype(topi::relu<float>)>;
                using ReluFuncPtr = decltype(&topi::relu<float>);

                template <typename ElementType>
                tvm::PackedFunc relu_builder(const std::unique_ptr<TVMInstance>& tvm_instance)
                {
                    throw ngraph_error(
                        "tvm_kernel::unary_elemwise_builder() instantiated with "
                        "unsupported ElementType");
                }

                template <>
                tvm::PackedFunc
                    relu_builder<float>(const std::unique_ptr<TVMInstance>& tvm_instance);
                using ReluBuilder = std::function<decltype(relu_builder<float>)>;

#define BUILD_TVM_RELU_FUNCTOR                                                                     \
    auto& functors = external_function->get_functors();                                            \
    auto& tvm_instance = external_function->get_tvm_instance();                                    \
    auto& tensor_data = external_function->get_tensor_data();                                      \
    tvm_kernel::ReluBuilder builder;                                                               \
    tvm_kernel::UnaryElemwiseKernel kernel;                                                        \
                                                                                                   \
    SELECT_KERNEL(builder, args[0].get_element_type(), tvm_kernel::relu_builder);                  \
    auto tvm_func = builder(tvm_instance);                                                         \
    SELECT_KERNEL(kernel, args[0].get_element_type(), tvm_kernel::unary_elemwise_kernel);          \
                                                                                                   \
    auto element_count = out[0].get_size();                                                        \
    auto& arg0_tensor = tensor_data[args[0].get_name()];                                           \
    auto& out0_tensor = tensor_data[out[0].get_name()];                                            \
                                                                                                   \
    auto functor = [&, tvm_func, kernel, element_count](CPURuntimeContext* ctx) {                  \
        kernel(tvm_instance, tvm_func, arg0_tensor, out0_tensor, element_count);                   \
    };                                                                                             \
    functors.emplace_back(functor);
            }
        }
    }
}
