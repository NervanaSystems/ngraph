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

#include <clang/CodeGen/ObjectFilePCHContainerOperations.h>
#include <clang/Driver/DriverDiagnostic.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/FrontendTool/Utils.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/LinkAllPasses.h>
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>

#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/Support/TargetSelect.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>

#include "compiler.hpp"

// TODO: Fix leaks

using namespace clang;
using namespace llvm;
using namespace llvm::opt;
using namespace std;

using namespace nervana::cpu;

static std::string GetExecutablePath(const char* Argv0)
{
    // This just needs to be some symbol in the binary; C++ doesn't
    // allow taking the address of ::main however.
    void* MainAddr = (void*)(intptr_t)GetExecutablePath;
    return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

execution_state::execution_state()
    : m_execution_engine{nullptr}
{
}

execution_state::~execution_state()
{
    // /// Take the LLVM context used by this action.
    // llvm::LLVMContext *takeLLVMContext();

    // if (m_execution_engine)
    // {
    //     m_execution_engine->runStaticConstructorsDestructors(true);
    // }
}

std::unique_ptr<llvm::Module> execution_state::compile(const string& source, const string& name)
{
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();

    // Prepare compilation arguments
    vector<const char*> args;
    args.push_back(name.c_str());

    // Prepare DiagnosticEngine
    DiagnosticOptions DiagOpts;
    TextDiagnosticPrinter* textDiagPrinter = new clang::TextDiagnosticPrinter(errs(), &DiagOpts);
    IntrusiveRefCntPtr<clang::DiagnosticIDs> pDiagIDs;
    DiagnosticsEngine* pDiagnosticsEngine =
        new DiagnosticsEngine(pDiagIDs, &DiagOpts, textDiagPrinter);

    // Create and initialize CompilerInstance
    std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());
    Clang->createDiagnostics();

    // Initialize CompilerInvocation
    CompilerInvocation::CreateFromArgs(
        Clang->getInvocation(), &args[0], &args[0] + args.size(), *pDiagnosticsEngine);

    // Infer the builtin include path if unspecified.
    if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
        Clang->getHeaderSearchOpts().ResourceDir.empty())
    {
        void* MainAddr = (void*)(intptr_t)GetExecutablePath;
        auto path = CompilerInvocation::GetResourcesPath(args[0], MainAddr);
        Clang->getHeaderSearchOpts().ResourceDir = path;
    }

    auto& HSO = Clang->getInvocation().getHeaderSearchOpts();
    // Add base toolchain-supplied header paths
    // Ideally one would use the Linux toolchain definition in clang/lib/Driver/ToolChains.h
    // But that's a private header and isn't part of the public libclang API
    // Instead of re-implementing all of that functionality in a custom toolchain
    // just hardcode the paths relevant to frequently used build/test machines for now
    HSO.AddPath("/usr/include/x86_64-linux-gnu", clang::frontend::System, false, false);
    HSO.AddPath("/usr/include", clang::frontend::System, false, false);
    // Add C++ standard library headers
    // Debian-like + GCC 4.8 libstdc++
    HSO.AddPath("/usr/include/x86_64-linux-gnu/c++/4.8", clang::frontend::System, false, false);
    HSO.AddPath("/usr/include/c++/4.8", clang::frontend::System, false, false);
    // Debian-like + GCC 5 libstdc++
    HSO.AddPath("/usr/include/x86_64-linux-gnu/c++/5", clang::frontend::System, false, false);
    HSO.AddPath("/usr/include/c++/5", clang::frontend::System, false, false);

    // Map code filename to a memoryBuffer
    StringRef source_ref(source);
    unique_ptr<MemoryBuffer> buffer = MemoryBuffer::getMemBufferCopy(source_ref);
    Clang->getInvocation().getPreprocessorOpts().addRemappedFile(name, buffer.get());

    // Create and execute action
    // CodeGenAction *compilerAction = new EmitLLVMOnlyAction();
    // CodeGenAction* compilerAction = new EmitAssemblyAction();
    CodeGenAction* compilerAction = new EmitCodeGenOnlyAction();
    Clang->ExecuteAction(*compilerAction);

    buffer.release();

    return compilerAction->takeModule();
}

bool execution_state::add_module(std::unique_ptr<llvm::Module>& module)
{
    if (module)
    {
        if (!m_execution_engine)
        {
            // auto mm = unique_ptr<RTDyldMemoryManager>(new method_resolver(this));
            m_execution_engine = llvm::EngineBuilder(move(module))
                                     .setEngineKind(llvm::EngineKind::JIT)
                                     .setOptLevel(llvm::CodeGenOpt::Aggressive)
                                     .setErrorStr(&jit_error)
                                     // .setUseMCJIT(true)
                                     // .setMCJITMemoryManager(std::move(mm))
                                     .create();

            if (!m_execution_engine)
            {
                return false;
            }
        }
    }
    else
    {
        return false;
    }

    return true;
}

void execution_state::finalize()
{
    if (m_execution_engine)
    {
        m_execution_engine->finalizeObject();
        m_execution_engine->runStaticConstructorsDestructors(false);
    }
    else
    {
        throw std::runtime_error(
            "Error in finalize: " +
            (jit_error.empty() ? "Could not create an execution engine" : jit_error));
    }
}
