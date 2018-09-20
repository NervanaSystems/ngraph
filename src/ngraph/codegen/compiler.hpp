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
#include <string>
#include <vector>

namespace ngraph
{
    namespace codegen
    {
        class Module;
        class Compiler;
        class CompilerCore;
    }
}

namespace clang
{
    class CompilerInstance;
    class CodeGenAction;
}

namespace llvm
{
    class Module;
}

class ngraph::codegen::Module
{
public:
    Module(std::unique_ptr<llvm::Module> module);
    ~Module();
    std::unique_ptr<llvm::Module> take_module();

private:
    std::unique_ptr<llvm::Module> m_module;
};

class ngraph::codegen::Compiler
{
public:
    Compiler();
    ~Compiler();
    void set_precompiled_header_source(const std::string& source);
    void add_header_search_path(const std::string& path);
    std::unique_ptr<ngraph::codegen::Module> compile(const std::string& source);
    std::unique_ptr<clang::CodeGenAction>& get_compiler_action() { return m_compiler_action; }
private:
    std::unique_ptr<clang::CodeGenAction> m_compiler_action;
    std::shared_ptr<CompilerCore> m_compiler_core;
    std::string m_precompiled_header_source;
    std::vector<std::string> m_header_search_paths;
};

class ngraph::codegen::CompilerCore
{
public:
    CompilerCore();
    ~CompilerCore();

    void set_debuginfo_enabled(bool state) { m_debuginfo_enabled = state; }
    bool is_debuginfo_enabled() { return m_debuginfo_enabled; }
    void set_precompiled_header_source(const std::string& source);
    const std::string& get_precompiled_header_source() const;
    void add_header_search_path(const std::string& path);

    std::unique_ptr<ngraph::codegen::Module>
        compile(std::unique_ptr<clang::CodeGenAction>& compiler_action, const std::string& source);
    std::string generate_pch(const std::string& source);
    void initialize();

private:
    std::unique_ptr<clang::CompilerInstance> m_compiler;
    bool m_debuginfo_enabled;
    bool m_enable_diag_output;
    bool m_enable_pass_report;
    std::string m_source_name;
    std::vector<std::string> m_extra_search_path_list;
    std::string m_precompiled_header_source;

    bool is_version_number(const std::string& path);
    std::string find_header_version(const std::string& path);
    std::string find_os_specific_path(const std::string& path);
    void configure_search_path();
    void load_headers_from_resource();
};
