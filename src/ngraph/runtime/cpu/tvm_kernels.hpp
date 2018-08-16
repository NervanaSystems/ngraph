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
        }
    }
}
