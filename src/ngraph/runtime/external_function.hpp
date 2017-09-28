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

namespace ngraph
{
    namespace runtime
    {
        class ExternalFunction
        {
            using FunctionMap = std::map<std::shared_ptr<Function>,ExternalFunction*>;

            using OpFunction  = std::function<void(const ngraph::Node*,
                                                   ExternalFunction*,
                                                   const std::vector<size_t>& inputs,
                                                   const std::vector<size_t>& outputs)>;
            using OpMap       = std::unordered_map<std::type_index, OpFunction>;

        public:
            ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                             const std::shared_ptr<FunctionMap>       function_map = nullptr,
                             bool                                     release_function = true,
                             bool                                     release_function_map = true);
            std::shared_ptr<ngraph::runtime::CallFrame> make_call_frame();
            std::shared_ptr<std::vector<std::shared_ptr<ngraph::runtime::Instruction>>>
                get_instructions()
            {
                return m_instructions;
            }

            // Release original function's resources
            void release_function() { m_function = nullptr; }

            // Release function map resources
            void release_function_map() { m_function_map = nullptr; }

        protected:
            void compile();
            void compile(std::shared_ptr<FunctionMap> function_map);

            std::shared_ptr<ngraph::Function>  m_function;
            bool                               m_release_function;
            bool                               m_is_compiled;
            size_t                             m_n_inputs;
            size_t                             m_n_outputs;
            std::shared_ptr<std::vector<std::shared_ptr<ngraph::runtime::Instruction>>>
                                               m_instructions;
            ngraph::descriptor::TensorViewPtrs m_temp_views;
            std::shared_ptr<FunctionMap>       m_function_map;
            bool                               m_release_function_map;

            static OpMap& get_op_map();
        };
    }
}
