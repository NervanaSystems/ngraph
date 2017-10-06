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
#include <string>

#include <llvm/ExecutionEngine/MCJIT.h> // forces JIT to link in
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/Option/Arg.h>

namespace nervana
{
    namespace cpu
    {
        class module;
        class execution_state;
    }
}

class nervana::cpu::module
{
public:
private:
    std::unique_ptr<llvm::Module> m_module;
};

class nervana::cpu::execution_state : public llvm::SectionMemoryManager
{
public:
    execution_state();
    ~execution_state();

    std::unique_ptr<llvm::Module> compile(const std::string& source, const std::string& name = "");

    bool add_module(std::unique_ptr<llvm::Module>&);

    void finalize();

    template <typename ftype>
    std::function<ftype> find_function(const std::string& func_name)
    {
        auto f = m_execution_engine->getPointerToNamedFunction(func_name);

        return f_cast<ftype>(f);
    }

private:
    llvm::ExecutionEngine* m_execution_engine;
    std::string jit_error;

    template <typename signature>
    std::function<signature> f_cast(void* f)
    {
        return static_cast<signature*>((signature*)f);
    }

    // class method_resolver : public llvm::RTDyldMemoryManager
    // {
    //     public:
    //     method_resolver(compiler* m);
    //     virtual uint64_t getSymbolAddress(const std::string &name) override;

    //     private:
    //     compiler*   m_Compiler;
    // };
};
