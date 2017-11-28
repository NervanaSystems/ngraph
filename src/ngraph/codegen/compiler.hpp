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

namespace ngraph
{
    namespace codegen
    {
        class module;
        class Compiler;
        class HeaderCache;
    }
}

namespace clang
{
    class HeaderSearchOptions;
    class CompilerInstance;
}

class ngraph::codegen::module
{
public:
private:
    std::unique_ptr<llvm::Module> m_module;
};

class ngraph::codegen::Compiler : public llvm::SectionMemoryManager
{
public:
    Compiler();
    ~Compiler();

    void set_precompiled_headers_enabled(bool state) { m_precompiled_headers_enabled = state; }
    bool is_precompiled_headers_enabled() { return m_precompiled_headers_enabled; }
    void set_debuginfo_enabled(bool state) { m_debuginfo_enabled = state; }
    bool is_debuginfo_enabled() { return m_debuginfo_enabled; }
    std::unique_ptr<llvm::Module> compile(const std::string& source);
    void set_precompiled_header_source(const std::string&) {}
private:
    std::unique_ptr<clang::CompilerInstance> m_compiler;
    bool m_precompiled_headers_enabled;
    bool m_debuginfo_enabled;
    std::string m_source_name;

    bool is_version_number(const std::string& path);
    void use_cached_files(std::unique_ptr<clang::CompilerInstance>& Clang);

    void load_header_search_path_from_resource();
    void load_headers_from_resource();
    void configure_search_path();
};

// // ----------------------------------------------------------------------------
// // Copyright 2017 Nervana Systems Inc.
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //      http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // ----------------------------------------------------------------------------

// #pragma once

// #include <functional>
// #include <memory>
// #include <string>

// #include <clang/CodeGen/CodeGenAction.h>

// #include <llvm/ExecutionEngine/MCJIT.h> // forces JIT to link in
// #include <llvm/ExecutionEngine/SectionMemoryManager.h>
// #include <llvm/Option/Arg.h>

// namespace ngraph
// {
//     namespace codegen
//     {
//         class module;
//         class Compiler;
//         class StaticCompiler;
//         class HeaderCache;
//     }
// }

// namespace clang
// {
//     class HeaderSearchOptions;
//     class CompilerInstance;
// }

// class ngraph::codegen::module
// {
// public:
// private:
//     std::unique_ptr<llvm::Module> m_module;
// };

// class ngraph::codegen::Compiler
// {
// public:
//     Compiler();
//     ~Compiler();
//     void set_precompiled_header_source(const std::string& source);
//     void add_header_search_path(const std::string& path);
//     std::unique_ptr<llvm::Module> compile(const std::string& source);
//     std::unique_ptr<clang::CodeGenAction>& get_compiler_action() { return compiler_action; }
// private:
//     std::unique_ptr<clang::CodeGenAction> compiler_action;
// };

// class ngraph::codegen::StaticCompiler : public llvm::SectionMemoryManager
// {
// public:
//     StaticCompiler();
//     ~StaticCompiler();

//     void set_debuginfo_enabled(bool state) { m_debuginfo_enabled = state; }
//     bool is_debuginfo_enabled() { return m_debuginfo_enabled; }
//     void set_precompiled_header_source(const std::string& source)
//     {
//         m_precomiled_header_source = source;
//     }
//     void add_header_search_path(const std::string& path);
//     void load_header_search_path_from_resource();
//     void load_headers_from_resource();
//     std::unique_ptr<llvm::Module> compile(std::unique_ptr<clang::CodeGenAction>& compiler_action,
//                                           const std::string& source);
//     void generate_pch(const std::string& source);

// private:
//     std::unique_ptr<clang::CompilerInstance> m_compiler;
//     bool m_precompiled_header_valid;
//     bool m_debuginfo_enabled;
//     std::string m_source_name;
//     std::vector<std::string> m_extra_search_path_list;
//     std::string m_pch_path;
//     std::string m_precomiled_header_source;
//     std::map<std::string, std::string> m_header_search_path_map;

//     bool is_version_number(const std::string& path);
//     void configure_search_path();
// };
