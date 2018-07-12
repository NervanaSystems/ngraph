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
#include <vector>

#include "ngraph/except.hpp"

namespace ngraph
{
    /// Error for ngraph assertion failure.
    class AssertionFailure : public ngraph_error
    {
    public:
        AssertionFailure(const std::string& what_arg)
            : ngraph_error(what_arg)
            , m_what(what_arg)
        {
        }

        AssertionFailure(const char* what_arg)
            : ngraph_error(what_arg)
            , m_what(what_arg)
        {
        }

        const char* what() const noexcept override { return m_what.c_str(); }
    private:
        std::string m_what;
    };

    class AssertionHelper
    {
    public:
        AssertionHelper(bool assertion_true,
                        std::string file,
                        int line,
                        std::vector<std::string> location_info = {})
            : m_assertion_true(assertion_true)
            , m_file(file)
            , m_line(line)
            , m_location_info(location_info)
        {
        }
        AssertionHelper(AssertionHelper&& other)
            : AssertionHelper(
                  other.m_assertion_true, other.m_file, other.m_line, other.m_location_info)
        {
            m_stream = std::move(other.m_stream);
        }
        ~AssertionHelper() noexcept(false);
        std::stringstream& get_stream() { return m_stream; }
    private:
        bool m_assertion_true;
        std::stringstream m_stream;
        std::string m_file;
        int m_line;
        std::vector<std::string> m_location_info;
    };
}

#define NGRAPH_ASSERT_WITH_LOC(cond, loc)                                                          \
    ::ngraph::AssertionHelper(cond, __FILE__, __LINE__, loc).get_stream()
#define NGRAPH_ASSERT(cond) ::ngraph::AssertionHelper(cond, __FILE__, __LINE__).get_stream()
