//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <map>

namespace ngraph
{
    namespace pass
    {
        class PassConfig;
        enum class CompilationMode
        {
            DEX,    // Fast compilation using precompiled kernels
            CODEGEN // Slower compilation for potentially faster code
        };
    }
}

class ngraph::pass::PassConfig
{
public:
    PassConfig(CompilationMode mode = CompilationMode::DEX);
    const std::map<std::string, bool>& get_enables() { return m_pass_enables; }
    void set_pass_enable(std::string name, bool enable);
    bool get_pass_enable(std::string name);
    const std::map<std::string, bool>& get_pass_attributes() { return m_pass_attributes; }
    void set_pass_attribute(std::string name, bool enable);
    bool get_pass_attribute(std::string name);
    CompilationMode get_compilation_mode() const { return m_compilation_mode; }
private:
    std::map<std::string, bool> m_pass_enables;
    std::map<std::string, bool> m_pass_attributes;
    CompilationMode m_compilation_mode;
};
