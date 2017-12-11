// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#include "ngraph/function.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/external_function.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_ExternalFunction;
            class CPU_Emitter;
            class CPU_CallFrame;

            using OpFunction = std::function<void(CPU_Emitter*,
                                                  const ngraph::Node*,
                                                  const std::vector<TensorViewWrapper>& inputs,
                                                  const std::vector<TensorViewWrapper>& outputs)>;

            using OpMap = std::unordered_map<std::type_index, OpFunction>;

            class CPU_ExternalFunction : public ngraph::runtime::ExternalFunction,
                                         public std::enable_shared_from_this<CPU_ExternalFunction>
            {
            public:
                CPU_ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                     bool release_function = true);
                std::shared_ptr<ngraph::runtime::CallFrame> make_call_frame();

            protected:
                void compile();

                EntryPoint m_compiled_function;

            private:
                std::unique_ptr<codegen::Compiler> compiler;
                std::unique_ptr<codegen::ExecutionEngine> execution_engine;
            };
        }
    }
}
