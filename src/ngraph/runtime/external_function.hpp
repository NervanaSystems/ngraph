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
#include "ngraph/runtime/tensor_view_info.hpp"

namespace ngraph
{
    namespace runtime
    {
        class CallFrame;

        class ExternalFunction
        {
        protected:
            ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                             bool release_function = true)
                : m_function(function)
                , m_release_function(release_function)
                , m_is_compiled(false)
            {
            }

            // Release original function's resources
            void release_function() { m_function = nullptr; }
        public:
            virtual ~ExternalFunction() {}
            virtual std::shared_ptr<CallFrame> make_call_frame() = 0;

        protected:
            std::shared_ptr<ngraph::Function> m_function;
            bool m_release_function;
            bool m_is_compiled;
        };
    }
}
