//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <memory>

#include <plaidml/plaidml++.h>

#include "ngraph/function.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/plaidml/plaidml_compiled_function.hpp"
#include "ngraph/runtime/plaidml/plaidml_config.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            struct Build;
            class Compiler;
        }
    }
}

// Compiles nGraph operation graphs (functions).
class ngraph::runtime::plaidml::Compiler final
{
public:
    Compiler(Config* config);

    std::shared_ptr<CompiledFunction> compile(std::shared_ptr<Function> func);

    void build(std::shared_ptr<Function> func, Build* build);

private:
    Config* m_config;
    ngraph::pass::Manager m_pass_manager;
};
