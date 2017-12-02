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

#include "ngraph/codegen/nvptx_compiler.hpp"
#include "ngraph/codegen/nvptx_execution_engine.hpp"
#include "ngraph/function.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/gpu/call_frame.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPUExternalFunction;
            class Emitter;
            class CallFrame;

            using OpFunction = std::function<void(Emitter*,
                                                  const ngraph::Node*,
                                                  const std::vector<TensorViewInfo>& inputs,
                                                  const std::vector<TensorViewInfo>& outputs)>;

            using OpMap = std::unordered_map<std::type_index, OpFunction>;

            class GPUExternalFunction : public ngraph::runtime::ExternalFunction,
                                        public std::enable_shared_from_this<GPUExternalFunction>
            {
            public:
                GPUExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                    bool release_function = true);
                std::shared_ptr<ngraph::runtime::CallFrame> make_call_frame();

            protected:
                void compile();

                EntryPoint m_compiled_function;

            private:
                std::unique_ptr<codegen::NVPTXCompiler> compiler;
                std::unique_ptr<codegen::NVPTXExecutionEngine> execution_engine;
            };
        }
    }
}
