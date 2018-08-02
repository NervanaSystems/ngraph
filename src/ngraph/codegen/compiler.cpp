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

#include <iostream>

#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/CodeGen/ObjectFilePCHContainerOperations.h>
#include <clang/Driver/DriverDiagnostic.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/FrontendTool/Utils.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ExecutionEngine/MCJIT.h> // forces JIT to link in
#include <llvm/IR/Module.h>
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

#include "header_resource.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

#if defined(__clang__)
#define IS_RTTI_ENABLED __has_feature(cxx_rtti)
#elif defined(__GNUC__)
#define IS_RTTI_ENABLED __GXX_RTTI
#else
// Unknown compiler so assume RTTI is enabled by default
#define IS_RTTI_ENABLED 1
#endif

#if IS_RTTI_ENABLED
#error "This source file interfaces with LLVM and Clang and must be compiled with RTTI disabled"
#endif

#define USE_BUILTIN

using namespace clang;
using namespace llvm;
using namespace std;
using namespace ngraph;

class CompilerInfo
{
public:
    string pch_file;
    shared_ptr<codegen::CompilerCore> compiler;
};

static unordered_map<string, CompilerInfo> s_compiler_info;

static class StaticHandler
{
public:
    StaticHandler() {}
    ~StaticHandler()
    {
        for (const auto& p : s_compiler_info)
        {
            file_util::remove_file(p.second.pch_file);
        }
    }
} s_static_init;

codegen::Module::Module(std::unique_ptr<llvm::Module> module)
    : m_module(move(module))
{
}

codegen::Module::~Module()
{
}

std::unique_ptr<llvm::Module> codegen::Module::take_module()
{
    return move(m_module);
}

codegen::Compiler::Compiler()
    : m_compiler_core{}
{
}

codegen::Compiler::~Compiler()
{
    m_compiler_action = nullptr;
    m_compiler_core = nullptr;
}

void codegen::Compiler::set_precompiled_header_source(const std::string& source)
{
    m_precompiled_header_source = source;
}

void codegen::Compiler::add_header_search_path(const std::string& path)
{
    m_header_search_paths.push_back(path);
}

std::unique_ptr<codegen::Module> codegen::Compiler::compile(const std::string& source)
{
    // lock_guard<mutex> lock(m_mutex);
    CompilerInfo& compiler_info = s_compiler_info[m_precompiled_header_source];
    if (!compiler_info.compiler)
    {
        compiler_info.compiler = make_shared<CompilerCore>();
        for (const string& path : m_header_search_paths)
        {
            compiler_info.compiler->add_header_search_path(path);
        }
        compiler_info.compiler->set_precompiled_header_source(m_precompiled_header_source);
    }
    auto rc = compiler_info.compiler->compile(m_compiler_action, source);
    return rc;
}

static std::string GetExecutablePath(const char* Argv0)
{
    // This just needs to be some symbol in the binary; C++ doesn't
    // allow taking the address of ::main however.
    void* MainAddr = reinterpret_cast<void*>(GetExecutablePath);
    return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

codegen::CompilerCore::CompilerCore()
    : m_debuginfo_enabled((std::getenv("NGRAPH_COMPILER_DEBUGINFO_ENABLE") != nullptr))
    , m_enable_diag_output((std::getenv("NGRAPH_COMPILER_DIAG_ENABLE") != nullptr))
    , m_enable_pass_report((std::getenv("NGRAPH_COMPILER_REPORT_ENABLE") != nullptr))
    , m_source_name("code.cpp")
{
    initialize();
}

void codegen::CompilerCore::initialize()
{
    m_extra_search_path_list.clear();

    InitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();

    // Prepare compilation arguments
    vector<const char*> args;
    args.push_back(m_source_name.c_str());

    // Inlining thresholds are forced to a very high value
    // to ensure all Eigen code gets properly inlined
    // This is for both Eigen strong and weak inlines
    args.push_back("-mllvm");
    args.push_back("-inline-threshold=1000000");
    if (m_enable_pass_report)
    {
        args.push_back("-Rpass-analysis=.*");
        args.push_back("-Rpass=.*");
        args.push_back("-Rpass-missed=.*");
    }
    // Prevent Eigen from using any LGPL3 code
    args.push_back("-DEIGEN_MPL2_ONLY");

    // Prepare DiagnosticEngine
    IntrusiveRefCntPtr<DiagnosticOptions> diag_options = new DiagnosticOptions();
    diag_options->ErrorLimit = 20;
    diag_options->ShowCarets = false;
    diag_options->ShowFixits = false;
    IntrusiveRefCntPtr<DiagnosticIDs> diag_id(new DiagnosticIDs());
    // create a diagnosetic buffer for errors caused by argument parsing
    TextDiagnosticBuffer* diag_buffer = new TextDiagnosticBuffer();
    DiagnosticsEngine diag_engine(diag_id, &*diag_options, diag_buffer);

    // Create and initialize CompilerInstance
    m_compiler = std::unique_ptr<CompilerInstance>(new CompilerInstance());

    // Initialize CompilerInvocation
    CompilerInvocation::CreateFromArgs(
        m_compiler->getInvocation(), &args[0], &args[0] + args.size(), diag_engine);

    DiagnosticConsumer* diag_consumer;
    if (m_enable_diag_output)
    {
        diag_consumer = new TextDiagnosticPrinter(errs(), &*diag_options);
    }
    else
    {
        diag_consumer = new IgnoringDiagConsumer();
    }
    // Create diagnostics after compiler invocation is created, otherwise report outputs do not get generated.
    m_compiler->createDiagnostics(diag_consumer);

    configure_search_path();

    // Language options
    // These are the C++ features needed to compile ngraph headers
    // and any dependencies like Eigen
    auto LO = m_compiler->getInvocation().getLangOpts();
    LO->CPlusPlus = 1;
    LO->CPlusPlus11 = 1;
    LO->Bool = 1;
    LO->Exceptions = 1;
    LO->CXXExceptions = 1;
    LO->WChar = 1;
    LO->RTTI = 1;
    // Enable OpenMP for Eigen
    LO->OpenMP = 1;
    LO->OpenMPUseTLS = 1;

    // CodeGen options
    auto& CGO = m_compiler->getInvocation().getCodeGenOpts();
    CGO.OptimizationLevel = 3;
    CGO.RelocationModel = "static";
    // CGO.CodeModel = "medium";
    CGO.ThreadModel = "posix";
    CGO.FloatABI = "hard";
    CGO.OmitLeafFramePointer = 1;
    CGO.VectorizeLoop = 1;
    CGO.VectorizeSLP = 1;
    CGO.CXAAtExit = 1;

    if (m_debuginfo_enabled)
    {
        CGO.setDebugInfo(codegenoptions::FullDebugInfo);
    }

    // Enable various target features
    auto& TO = m_compiler->getInvocation().getTargetOpts();
    TO.CPU = sys::getHostCPUName();

    // Flush out any errors from clang/llvm arg parsing.
    diag_buffer->FlushDiagnostics(m_compiler->getDiagnostics());
}

codegen::CompilerCore::~CompilerCore()
{
    // This is causing a segfault after program terminates
    // will address later
    if (m_compiler)
    {
        // PreprocessorOptions& preprocessor_options =
        //     m_compiler->getInvocation().getPreprocessorOpts();
        // for (auto& x : preprocessor_options.RemappedFileBuffers)
        // {
        //     delete x.second;
        // }
        m_compiler = nullptr;
    }
}

bool codegen::CompilerCore::is_version_number(const string& path)
{
    bool rc = true;
    vector<string> tokens = split(path, '.');
    for (string s : tokens)
    {
        for (char c : s)
        {
            if (!isdigit(c))
            {
                rc = false;
            }
        }
    }
    return rc;
}

void codegen::CompilerCore::add_header_search_path(const string& p)
{
    vector<string> paths = split(p, ';');
    for (const string& path : paths)
    {
        if (!contains(m_extra_search_path_list, path))
        {
            m_extra_search_path_list.push_back(path);
            HeaderSearchOptions& hso = m_compiler->getInvocation().getHeaderSearchOpts();
            hso.AddPath(path, clang::frontend::System, false, false);
        }
    }
}

std::unique_ptr<codegen::Module>
    codegen::CompilerCore::compile(std::unique_ptr<clang::CodeGenAction>& m_compiler_action,
                                   const string& source)
{
    PreprocessorOptions& preprocessor_options = m_compiler->getInvocation().getPreprocessorOpts();

    preprocessor_options.RetainRemappedFileBuffers = true;

    CompilerInfo& compiler_info = s_compiler_info[m_precompiled_header_source];
    if (!m_precompiled_header_source.empty() && compiler_info.pch_file.empty())
    {
        compiler_info.pch_file = generate_pch(m_precompiled_header_source);
    }
    if (!compiler_info.pch_file.empty())
    {
        // Preprocessor options
        preprocessor_options.ImplicitPCHInclude = compiler_info.pch_file;
        preprocessor_options.DisablePCHValidation = 0;
    }

    // Clear warnings and errors
    m_compiler->getDiagnosticClient().clear();

    // Map code filename to a memoryBuffer
    StringRef source_ref(source);
    unique_ptr<MemoryBuffer> buffer = MemoryBuffer::getMemBufferCopy(source_ref);
    preprocessor_options.RemappedFileBuffers.push_back({m_source_name, buffer.get()});

    // Create and execute action
    m_compiler_action.reset(new EmitCodeGenOnlyAction());
    std::unique_ptr<llvm::Module> rc;
    bool reinitialize = false;
    if (m_compiler->ExecuteAction(*m_compiler_action) == true)
    {
        rc = m_compiler_action->takeModule();
    }
    else
    {
        reinitialize = true;
    }

    buffer.release();

    preprocessor_options.RemappedFileBuffers.pop_back();

    unique_ptr<codegen::Module> result;
    if (rc)
    {
        result = move(unique_ptr<codegen::Module>(new codegen::Module(move(rc))));
    }
    else
    {
        result = move(unique_ptr<codegen::Module>(nullptr));
    }

    if (reinitialize)
    {
        codegen::CompilerCore::initialize();
    }

    return result;
}

string codegen::CompilerCore::generate_pch(const string& source)
{
    PreprocessorOptions& preprocessor_options = m_compiler->getInvocation().getPreprocessorOpts();
    string pch_path = file_util::tmp_filename();
    m_compiler->getFrontendOpts().OutputFile = pch_path;

    // Map code filename to a memoryBuffer
    StringRef source_ref(source);
    unique_ptr<MemoryBuffer> buffer = MemoryBuffer::getMemBufferCopy(source_ref);
    preprocessor_options.RemappedFileBuffers.push_back({m_source_name, buffer.get()});

    // Create and execute action
    clang::GeneratePCHAction* compilerAction = new clang::GeneratePCHAction();
    if (m_compiler->ExecuteAction(*compilerAction) == false)
    {
        file_util::remove_file(pch_path);
        pch_path = "";
    }
    else
    {
        s_compiler_info[source].pch_file = pch_path;
    }

    buffer.release();
    preprocessor_options.RemappedFileBuffers.pop_back();

    delete compilerAction;

    return pch_path;
}

void codegen::CompilerCore::configure_search_path()
{
#ifdef USE_BUILTIN
    load_headers_from_resource();
#endif

#if defined(__APPLE__)
    add_header_search_path(EIGEN_HEADERS_PATH);
    add_header_search_path(MKLDNN_HEADERS_PATH);
    add_header_search_path(TBB_HEADERS_PATH);
    add_header_search_path(NGRAPH_HEADERS_PATH);
    add_header_search_path(INSTALLED_HEADERS_PATH);
    add_header_search_path(CLANG_BUILTIN_HEADERS_PATH);

    add_header_search_path("/Library/Developer/CommandLineTools/usr/include/c++/v1");
#else
    // Add base toolchain-supplied header paths
    // Ideally one would use the Linux toolchain definition in clang/lib/Driver/ToolChains.h
    // But that's a private header and isn't part of the public libclang API
    // Instead of re-implementing all of that functionality in a custom toolchain
    // just hardcode the paths relevant to frequently used build/test machines for now
    add_header_search_path(CLANG_BUILTIN_HEADERS_PATH);

    string header_version = find_header_version("/usr/include/c++");
    string os_specific_path =
        find_os_specific_path(file_util::path_join("/usr/include/c++", header_version));

    // /usr/include/c++/7
    add_header_search_path(file_util::path_join("/usr/include/c++/", header_version));

    // /usr/include/x86_64-linux-gnu/c++/7
    add_header_search_path(
        file_util::path_join("/usr/include/x86_64-linux-gnu/c++/", header_version));

    add_header_search_path(
        file_util::path_join("/usr/lib/gcc/x86_64-linux-gnu/", header_version, "/include"));
    add_header_search_path("/usr/local/include");
    add_header_search_path(
        file_util::path_join("/usr/include/c++/", header_version, os_specific_path));
    add_header_search_path(
        file_util::path_join("/usr/lib/gcc/x86_64-linux-gnu/", header_version, "/include-fixed"));
    add_header_search_path("/usr/include/x86_64-linux-gnu");
    add_header_search_path("/usr/include");

    add_header_search_path(EIGEN_HEADERS_PATH);
    add_header_search_path(MKLDNN_HEADERS_PATH);
    add_header_search_path(TBB_HEADERS_PATH);
    add_header_search_path(NGRAPH_HEADERS_PATH);
    add_header_search_path(INSTALLED_HEADERS_PATH);
#endif

#ifdef CUDA_HEADER_PATHS
    // Only needed for GPU backend
    add_header_search_path(CUDA_HEADER_PATHS);
#endif

#ifdef CUDNN_HEADER_PATHS
    // Only needed for GPU backend
    add_header_search_path(CUDNN_HEADER_PATHS);
#endif

#ifdef NGRAPH_DISTRIBUTED
    add_header_search_path(MPI_HEADER_PATH);
#endif
}

void codegen::CompilerCore::load_headers_from_resource()
{
    const string builtin_root = "";
    HeaderSearchOptions& hso = m_compiler->getInvocation().getHeaderSearchOpts();
    PreprocessorOptions& preprocessor_options = m_compiler->getInvocation().getPreprocessorOpts();
    // for (const string& search_path : builtin_search_paths)
    // {
    //     string builtin = builtin_root + search_path;
    //     hso.AddPath(builtin, clang::frontend::System, false, false);
    // }
    for (const pair<string, string>& header_info : builtin_headers)
    {
        string absolute_path = header_info.first;
        string builtin = builtin_root + absolute_path;
        std::unique_ptr<llvm::MemoryBuffer> mb(
            llvm::MemoryBuffer::getMemBuffer(header_info.second, builtin));
        preprocessor_options.addRemappedFile(builtin, mb.release());
    }
}

void codegen::CompilerCore::set_precompiled_header_source(const std::string& source)
{
    m_precompiled_header_source = source;
}

const string& codegen::CompilerCore::get_precompiled_header_source() const
{
    return m_precompiled_header_source;
}

string codegen::CompilerCore::find_header_version(const string& path)
{
    vector<string> directories;
    string rc;
    auto f = [&](const std::string& file, bool is_dir) {
        if (is_dir)
        {
            directories.push_back(file);
        }
    };
    file_util::iterate_files(path, f);
    for (const string& dir : directories)
    {
        string dir_name = file_util::get_file_name(dir);
        if (is_version_number(dir_name))
        {
            rc = dir_name;
            break;
        }
    }
    return rc;
}

string codegen::CompilerCore::find_os_specific_path(const string& path)
{
    string rc;
    auto f = [&](const std::string& file, bool is_dir) {
        if (is_dir)
        {
            const string prefix = "x86_64-";
            const string suffix = "-linux";
            string path = file_util::get_file_name(file);
            if (path.size() > (prefix.size() + suffix.size()) &&
                path.compare(0, prefix.size(), prefix) == 0 &&
                path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0)
            {
                rc = path.substr(prefix.size(), path.size() - prefix.size() - suffix.size());
                rc = prefix + rc + suffix;
            }
        }
    };
    file_util::iterate_files(path, f);
    return rc;
}
