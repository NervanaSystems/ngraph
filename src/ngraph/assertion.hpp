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

#include <exception>
#include <sstream>
#include <vector>

#include "ngraph/except.hpp"

namespace ngraph
{
    /// Base class for ngraph assertion failures.
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

    template <class T>
    class AssertionHelper
    {
    public:
        AssertionHelper(const std::string& file,
                        int line,
                        const std::string& assertion_expression = "",
                        const std::string& location_info = "")
            : m_file(file)
            , m_line(line)
            , m_assertion_expression(assertion_expression)
            , m_location_info(location_info)
        {
        }
        ~AssertionHelper() noexcept(false)
        {
            if (!std::uncaught_exception())
            {
                std::stringstream ss;
                if (!m_location_info.empty())
                {
                    ss << m_location_info << ":" << std::endl;
                }

                if (m_assertion_expression.empty())
                {
                    ss << "Failure ";
                }
                else
                {
                    ss << "Assertion '" << m_assertion_expression << "' failed ";
                }

                ss << "at " << m_file << ":" << m_line << ":" << std::endl;

                std::string explanation = m_stream.str();
                if (explanation.empty())
                {
                    explanation = "(no explanation given)";
                }
                ss << explanation;

                throw T(ss.str());
            }
        }
        std::ostream& get_stream() { return m_stream; }
    private:
        std::stringstream m_stream;
        std::string m_file;
        int m_line;
        std::string m_assertion_expression;
        std::string m_location_info;
    };

    class DummyAssertionHelper
    {
    public:
        std::ostream& get_stream() { return m_stream; }
    private:
        std::stringstream m_stream;
    };
}

#define NGRAPH_ASSERT_STREAM_WITH_LOC(T, cond, loc)                                                \
    (cond ? ::ngraph::DummyAssertionHelper().get_stream()                                          \
          : ::ngraph::AssertionHelper<T>(__FILE__, __LINE__, #cond, loc).get_stream())
#define NGRAPH_ASSERT_STREAM(T, cond)                                                              \
    (cond ? ::ngraph::DummyAssertionHelper().get_stream()                                          \
          : ::ngraph::AssertionHelper<T>(__FILE__, __LINE__, #cond).get_stream())
#define NGRAPH_FAIL_STREAM_WITH_LOC(T, loc)                                                        \
    ::ngraph::AssertionHelper<T>(__FILE__, __LINE__, "", loc).get_stream()
#define NGRAPH_FAIL_STREAM(T) ::ngraph::AssertionHelper<T>(__FILE__, __LINE__).get_stream()

#define NGRAPH_ASSERT_WITH_LOC(cond, loc)                                                          \
    NGRAPH_ASSERT_STREAM_WITH_LOC(::ngraph::AssertionFailure, cond, loc)
#define NGRAPH_ASSERT(cond) NGRAPH_ASSERT_STREAM(::ngraph::AssertionFailure, cond)
#define NGRAPH_FAIL_WITH_LOC(loc) NGRAPH_FAIL_STREAM_WITH_LOC(::ngraph::AssertionFailure, loc)
#define NGRAPH_FAIL() NGRAPH_FAIL_STREAM(::ngraph::AssertionFailure)
