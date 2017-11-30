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

#include "ngraph/codegen/execution_engine.hpp"
#include "ngraph/runtime/manager.hpp"

namespace ngraph
{
    class Function;

    namespace runtime
    {
        class ExternalFunction;

        namespace interpreter
        {
            /// @brief Transformer for the interpreted backend
            class INT_Manager : public Manager
            {
            protected:
                ngraph::codegen::ExecutionEngine exec_state;

            public:
                virtual std::shared_ptr<Backend> allocate_backend() override;

                virtual std::shared_ptr<ngraph::runtime::ExternalFunction>
                    compile(const std::shared_ptr<ngraph::Function>& fun) override;

                static Factory factory;
            };
        };
    }
}
