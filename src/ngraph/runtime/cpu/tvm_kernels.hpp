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
#include <topi/broadcast.h>
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
                template <typename ElementType>
                void binary_elemwise_compute(const std::unique_ptr<TVMInstance>& tvm_instance,
                                             const tvm::runtime::PackedFunc& func,
                                             void* input0,
                                             void* input1,
                                             void* output,
                                             size_t count)
                {
                    std::cout << "divide compute" << std::endl;
                }
                template <>
                void
                    binary_elemwise_compute<float>(const std::unique_ptr<TVMInstance>& tvm_instance,
                                                   const tvm::runtime::PackedFunc& func,
                                                   void* input0,
                                                   void* input1,
                                                   void* output,
                                                   size_t count);

                template <typename ElementType>
                tvm::PackedFunc build_divide(const std::unique_ptr<TVMInstance>& tvm_instance)
                {
                    std::cout << "divide build" << std::endl;
                }

                template <>
                tvm::PackedFunc
                    build_divide<float>(const std::unique_ptr<TVMInstance>& tvm_instance);
            }
        }
    }
}
