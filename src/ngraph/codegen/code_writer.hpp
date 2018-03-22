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

#pragma once

#include <sstream>
#include <string>

namespace ngraph
{
    namespace codegen
    {
        class CodeWriter;
    }
}

class ngraph::codegen::CodeWriter
{
public:
    CodeWriter();
    std::string get_code() const;

    void operator+=(const std::string&);

    size_t indent;

    template <typename T>
    friend CodeWriter& operator<<(CodeWriter& out, const T& obj)
    {
        std::stringstream ss;
        ss << obj;

        for (char c : ss.str())
        {
            if (c == '\n')
            {
                out.m_pending_indent = true;
            }
            else
            {
                if (out.m_pending_indent)
                {
                    out.m_pending_indent = false;
                    for (size_t i = 0; i < out.indent; i++)
                    {
                        out.m_ss << "    ";
                    }
                }
            }
            out.m_ss << c;
        }

        return out;
    }

    std::string generate_temporary_name(std::string prefix = "tempvar");

    void block_begin(std::string block_prefix = "")
    {
        *this << "{" << block_prefix << "\n";
        indent++;
    }

    void block_end(std::string block_suffix = "")
    {
        indent--;
        *this << "}" << block_suffix << "\n";
    }

private:
    std::stringstream m_ss;
    bool m_pending_indent;
    size_t m_temporary_name_count;
};
