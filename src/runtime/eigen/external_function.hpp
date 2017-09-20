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

#include "function.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace eigen
        {
            class ExternalFunction
            {
            public:
                ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                 bool release_function = true);
                std::shared_ptr<ngraph::runtime::CallFrame> make_call_frame();
                std::shared_ptr<std::vector<std::shared_ptr<ngraph::runtime::Instruction>>>
                    get_instructions()
                {
                    return m_instructions;
                }

                // Release original function's resources
                void release_function() { m_function = nullptr; }

            protected:
                void compile();

                std::shared_ptr<ngraph::Function> m_function;
                bool                              m_release_function;
                bool                              m_is_compiled;
                size_t                            m_n_inputs;
                size_t                            m_n_outputs;
                std::shared_ptr<std::vector<std::shared_ptr<ngraph::runtime::Instruction>>>
                                                                             m_instructions;
                std::vector<std::shared_ptr<ngraph::descriptor::TensorView>> m_temp_views;

                static std::unordered_map<std::type_index,
                                          std::function<void(ngraph::Node*,
                                                             ExternalFunction*,
                                                             const std::vector<size_t>& inputs,
                                                             const std::vector<size_t>& outputs)>>&
                    get_op_map();
            };
        }
    }
}
