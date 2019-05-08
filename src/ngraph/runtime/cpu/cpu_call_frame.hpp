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

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/runtime/allocator.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"
#include "ngraph/runtime/tensor.hpp"

class CPURuntimeContextCG;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class CPU_Debugger;

            using InitContextFuncTy = CPURuntimeContextCG*();
            using DestroyContextFuncTy = void(CPURuntimeContextCG*);
            using EntryPointTy = void(void** inputs,
                                      void** outputs,
                                      CPURuntimeContext* ctx,
                                      CPURuntimeContextCG* cg_ctx);

            using InitContextFuncCG = std::function<InitContextFuncTy>;
            using DestroyContextFuncCG = std::function<DestroyContextFuncTy>;
            using EntryPoint = std::function<EntryPointTy>;

            // Compile and execute graphs
            class CPU_CallFrame
            {
            public:
                friend class CPU_Debugger;

                CPU_CallFrame(std::shared_ptr<CPU_ExternalFunction> external_function,
                              InitContextFuncCG compiled_init_ctx_func,
                              DestroyContextFuncCG compiled_destroy_ctx_func,
                              EntryPoint compiled_function,
                              runtime::Allocator* allocator);
                ~CPU_CallFrame();

                /// \brief Invoke the function with values matching the signature of the function.
                ///
                /// Tuples will be expanded into their tensor views to build the call frame.
                void call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs);

                void propagate_layouts(const std::vector<std::shared_ptr<runtime::Tensor>>& tvs,
                                       const LayoutDescriptorPtrs& layouts) const;

                void setup_runtime_context(runtime::Allocator* allocator);
                void setup_cg_runtime_context();
                void cleanup_runtime_context();

            protected:
                CPU_CallFrame(const CPU_CallFrame&) = delete;
                CPU_CallFrame(CPU_CallFrame&&) = delete;
                CPU_CallFrame& operator=(const CPU_CallFrame&) = delete;

                void inner_call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                const std::vector<std::shared_ptr<runtime::Tensor>>& inputs,
                                const size_t id,
                                const bool disable_caching = true);

                std::shared_ptr<CPU_ExternalFunction> m_external_function;

                std::mutex m_mutex;
                std::condition_variable m_cv;
                volatile size_t m_num_ctx_available = 0;
                size_t m_prev_ctx = 0;
                size_t m_num_ctx = 1;
                std::unordered_map<size_t, bool> m_id_pool;
                std::vector<CPURuntimeContext*> m_ctx_vec;

                /* Codegen specific */

                /// Function that initializes the context used in codegen mode.
                InitContextFuncCG m_compiled_init_ctx_func;

                /// Function that destroys the context used in codegen mode.
                DestroyContextFuncCG m_compiled_destroy_ctx_func;

                EntryPoint m_compiled_function;

                /// Execution context used in codegen mode.
                CPURuntimeContextCG* cg_ctx = nullptr;
            };
        }
    }
}
