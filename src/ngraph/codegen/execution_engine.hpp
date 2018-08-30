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

#include <functional>
#include <memory>

#include "ngraph/codegen/compiler.hpp"

namespace ngraph
{
    namespace codegen
    {
        class ExecutionEngine;
    }
}

namespace llvm
{
    class Module;
    class ExecutionEngine;
}

class ngraph::codegen::ExecutionEngine
{
public:
    ExecutionEngine();
    ~ExecutionEngine();

    bool add_module(std::unique_ptr<ngraph::codegen::Module>& module);
    void finalize();

    template <typename ftype>
    std::function<ftype> find_function(const std::string& func_name)
    {
        return f_cast<ftype>(get_pointer_to_named_function(func_name));
    }

private:
    std::unique_ptr<llvm::ExecutionEngine> m_execution_engine;
    std::string m_jit_error;

    void* get_pointer_to_named_function(const std::string& func_name);
    template <typename signature>
    std::function<signature> f_cast(void* f)
    {
        return static_cast<signature*>(reinterpret_cast<signature*>(f));
    }
};
