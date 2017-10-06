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

#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/function.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ngvm
        {
            class Instruction;

            class ExternalFunction : public ngraph::runtime::ExternalFunction
            {
                using FunctionMap = std::unordered_map<std::shared_ptr<Function>,
                                                       std::shared_ptr<ExternalFunction>>;

                using OpFunction = std::function<void(const ngraph::Node*,
                                                      ExternalFunction*,
                                                      FunctionMap&,
                                                      const std::vector<TensorViewInfo>& inputs,
                                                      const std::vector<TensorViewInfo>& outputs)>;
                using OpMap = std::unordered_map<std::type_index, OpFunction>;

            public:
                ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                 bool release_function = true);
                std::shared_ptr<ngraph::runtime::CallFrame> make_call_frame();
                virtual std::shared_ptr<ngraph::runtime::CallFrame>
                    make_call_frame(FunctionMap& function_map);
                std::shared_ptr<std::vector<std::shared_ptr<Instruction>>> get_instructions()
                {
                    return m_instructions;
                }

                // Release original function's resources
                void release_function() { m_function = nullptr; }
            protected:
                void compile();
                void compile(FunctionMap& function_map);

                size_t m_n_inputs;
                size_t m_n_outputs;
                std::shared_ptr<std::vector<std::shared_ptr<Instruction>>> m_instructions;
                ngraph::descriptor::TensorViewPtrs m_temp_views;

                static OpMap& get_op_map();
            };
        }
    }
}
