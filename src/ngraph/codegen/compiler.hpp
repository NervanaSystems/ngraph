#pragma once

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
