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
        class StaticCompiler;
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

class ngraph::codegen::Compiler
{
public:
    Compiler();
    ~Compiler();
    std::unique_ptr<llvm::Module> compile(const std::string& source);

private:
};

class ngraph::codegen::StaticCompiler : public llvm::SectionMemoryManager
{
public:
    StaticCompiler();
    ~StaticCompiler();

    void set_precompiled_headers_enabled(bool state) { m_precompiled_headers_enabled = state; }
    bool is_precompiled_headers_enabled() { return m_precompiled_headers_enabled; }
    void set_debuginfo_enabled(bool state) { m_debuginfo_enabled = state; }
    bool is_debuginfo_enabled() { return m_debuginfo_enabled; }
    std::unique_ptr<llvm::Module> compile(const std::string& source);

private:
    std::unique_ptr<clang::CompilerInstance> m_compiler;
    bool m_precompiled_headers_enabled;
    bool m_debuginfo_enabled;
    std::string m_source_name;

    bool is_version_number(const std::string& path);
    void add_header_search_path(clang::HeaderSearchOptions& hso, const std::string& path);
    void use_cached_files(std::unique_ptr<clang::CompilerInstance>& Clang);
};

class ngraph::codegen::HeaderCache
{
public:
    bool is_valid() const { return m_headers_valid; }
    bool set_valid() { return m_headers_valid = true; }
    void add_path(const std::string& path) { m_include_paths.push_back(path); }
    void add_file(const std::string& path, std::unique_ptr<llvm::MemoryBuffer>& code)
    {
        m_headers.insert(std::make_pair(path, std::move(code)));
    }
    const std::map<std::string, std::unique_ptr<llvm::MemoryBuffer>>& get_header_map() const
    {
        return m_headers;
    }
    const std::vector<std::string>& get_include_paths() const { return m_include_paths; }
private:
    std::map<std::string, std::unique_ptr<llvm::MemoryBuffer>> m_headers;
    std::vector<std::string> m_include_paths;
    bool m_headers_valid;
};
