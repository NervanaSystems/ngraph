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

#include <exception>
#include <sstream>
#include <vector>

#include "ngraph/except.hpp"

namespace ngraph
{
    /// Base class for ngraph assertion failure exceptions.
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

    ///
    /// Helper class for failed assertions. Callers should not instantiate this class directly.
    /// This class is meant to be wrapped with a macro like NGRAPH_ASSERT. This class provides
    /// two main facilities: (1) an ostream accessible via get_stream(), to which a detailed
    /// error explanation can be written; and (2) throws an exception of type T when the
    /// AssertionHelper is destructed.
    ///
    ///
    /// Typical usage is via a wrapper around the NGRAPH_ASSERT_STREAM macro:
    ///
    ///    class MyException : public AssertionFailure;
    ///
    ///    #define MY_ASSERT(cond) NGRAPH_ASSERT_STREAM(::ngraph::MyException, cond)
    ///
    ///    ...
    ///
    ///    MY_ASSERT(42 != 43) << "Uh-oh. " << 42 << " is not " << 43 << ".";
    ///
    /// If the assertion fails, it will throw a CompileError exception with a what() string of:
    ///
    ///   Assertion '42 != 43' failed at foo.cpp:123:
    ///   Uh-oh. 42 is not 43.
    ///
    ///
    /// AssertionHelper also provides support for tagging the exception with a "location" string,
    /// reflecting things like the op that was being processed when the error occurred. For
    /// example:
    ///
    ///   class CompileError : public AssertionFailure;
    ///
    ///   #define COMPILE_ASSERT(node,cond)                                                  \
    ///      NGRAPH_ASSERT_STREAM_WITH_LOC(::ngraph::CompileError, cond,                     \
    ///                                    "While compiling node " + node->name())
    ///
    ///   ...
    ///
    ///   COMPILE_ASSERT(node, node->get_users().size != 0) << "Node has no users";
    ///
    /// If the assertion fails, it will throw a CompileError exception with a what() string
    /// similar to:
    ///
    ///   While compiling node Add_123:
    ///   Assertion 'node->get_users().size != 0' failed at foo.cpp:123:
    ///   Node has no users
    ///
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
            // If stack unwinding is already in progress, do not double-throw.
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
        /// Returns an ostream to which additional error details can be written. The returned
        /// stream has the lifetime of the AssertionHelper.
        std::ostream& get_stream() { return m_stream; }
    private:
        std::stringstream m_stream;
        std::string m_file;
        int m_line;
        std::string m_assertion_expression;
        std::string m_location_info;
    };

    ///
    /// Class that returns a dummy ostream to absorb error strings for non-failed assertions.
    /// This is cheaper to construct than AssertionHelper, so the macros will produce a
    /// DummyAssertionHelper in lieu of an AssertionHelper if the condition is true.
    ///
    class DummyAssertionHelper
    {
    public:
        /// Returns an ostream to which additional error details can be written. Anything written
        /// to this stream will be ignored. The returned stream has the lifetime of the
        /// DummyAssertionHelper.
        std::ostream& get_stream() { return m_stream; }
    private:
        std::stringstream m_stream;
    };
}

/// Asserts condition "cond" with an exception class of "T", at location "loc".
#define NGRAPH_ASSERT_STREAM_WITH_LOC(T, cond, loc)                                                \
    (cond ? ::ngraph::DummyAssertionHelper().get_stream()                                          \
          : ::ngraph::AssertionHelper<T>(__FILE__, __LINE__, #cond, loc).get_stream())
/// Asserts condition "cond" with an exception class of "T", and no location specified.
#define NGRAPH_ASSERT_STREAM(T, cond)                                                              \
    (cond ? ::ngraph::DummyAssertionHelper().get_stream()                                          \
          : ::ngraph::AssertionHelper<T>(__FILE__, __LINE__, #cond).get_stream())
/// Fails unconditionally with an exception class of "T", at location "loc".
#define NGRAPH_FAIL_STREAM_WITH_LOC(T, loc)                                                        \
    ::ngraph::AssertionHelper<T>(__FILE__, __LINE__, "", loc).get_stream()
/// Fails unconditionally with an exception class of "T", and no location specified.
#define NGRAPH_FAIL_STREAM(T) ::ngraph::AssertionHelper<T>(__FILE__, __LINE__).get_stream()

#define NGRAPH_ASSERT(cond) NGRAPH_ASSERT_STREAM(::ngraph::AssertionFailure, cond)
#define NGRAPH_FAIL() NGRAPH_FAIL_STREAM(::ngraph::AssertionFailure)
