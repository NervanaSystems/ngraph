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

#include "ngraph/function.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/interpreter/int_call_frame.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class ExternalFunction : public ngraph::runtime::ExternalFunction,
                                     public std::enable_shared_from_this<ExternalFunction>
            {
            public:
                ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                std::shared_ptr<INT_Backend> backend,
                                 bool release_function = true);
                std::shared_ptr<ngraph::runtime::CallFrame> make_call_frame();

            protected:
                std::shared_ptr<ngraph::Function> m_function;
                std::shared_ptr<INT_Backend> m_backend;
                void compile();
            };
        }
    }
}
