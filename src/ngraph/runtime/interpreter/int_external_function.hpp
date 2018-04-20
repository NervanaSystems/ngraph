/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <memory>

#include "ngraph/function.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class INT_CallFrame;

            class ExternalFunction
            {
            public:
                ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                 bool release_function = false);
                std::shared_ptr<INT_CallFrame> make_call_frame();

            protected:
                void compile();
                void release_function() { m_function = nullptr; }
                std::shared_ptr<ngraph::Function> m_function;
                bool m_release_function;
                bool m_is_compiled;
                bool m_timing;
            };
        }
    }
}
