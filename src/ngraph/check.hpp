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

#include <exception>
#include <sstream>
#include <vector>

#include "ngraph/except.hpp"

namespace ngraph
{
    static inline std::ostream& write_all_to_stream(std::ostream& str) { return str; }
    template <typename T, typename... TS>
    static inline std::ostream& write_all_to_stream(std::ostream& str, const T& arg, TS&&... args)
    {
        return write_all_to_stream(str << arg, args...);
    }

    struct CheckLocInfo
    {
        const char* file;
        int line;
        const char* check_string;
    };

    /// Base class for check failure exceptions.
    class CheckFailure : public ngraph_error
    {
    public:
        CheckFailure(const CheckLocInfo& check_loc_info,
                     const std::string& context_info,
                     const std::string& explanation)
            : ngraph_error(make_what(check_loc_info, context_info, explanation))
        {
        }

    private:
        static std::string make_what(const CheckLocInfo& check_loc_info,
                                     const std::string& context_info,
                                     const std::string& explanation)
        {
            std::stringstream ss;
            ss << "Check '" << check_loc_info.check_string << "' failed at " << check_loc_info.file
               << ":" << check_loc_info.line;
            if (!context_info.empty())
            {
                ss << ":" << std::endl << context_info;
            }
            if (!explanation.empty())
            {
                ss << ":" << std::endl << explanation;
            }
            ss << std::endl;
            return ss.str();
        }
    };
}

//
// Helper macro for defining custom check macros, which throw custom exception classes and provide
// useful context information (the check condition, source filename, line number, and any domain-
// specific context information [e.g., a summary of the node that was being processed at the time
// of the check]).
//
// For example (actually implemented in node.cpp), let's say we want to define a macro for
// checking conditions during node validation, usable as follows:
//
//    NODE_VALIDATION_CHECK(node_being_checked,
//                          node_being_checked->get_input_shape(0).size() == 1,
//                          "Node must have an input rank of 1, but got ",
//                          node_being_checked->get_input_shape(0).size(), ".");
//
// In case of failure, this will throw an exception of type NodeValidationFailure with a what()
// string something like:
//
//      Check 'node_being_checked->get_input_shape(0).size() == 1' failed at foo.cpp:123:
//      While validating node 'Broadcast[Broadcast_10](Reshape_9: float{1,3,4,5}) -> (??)':
//      Node must have an input of rank 1, but got 2.
//
// To implement this, he first step is to define a subclass of CheckFailure (let's say it's called
// MyFailure), which must have a constructor of the form:
//
//      MyFailure(const CheckLocInfo& check_loc_info,
//                T context_info, // "T" can be any type; you'll supply a function to convert "T"
//                                // to std::string
//                const std::string& explanation)
//
// Here, we define a custom class for node validation failures as follows:
//
//    static std::string node_validation_failure_loc_string(const Node* node)
//    {
//        std::stringstream ss;
//        ss << "While validating node '" << *node << "'";
//        return ss.str();
//    }
//
//    class NodeValidationFailure : public CheckFailure
//    {
//    public:
//        NodeValidationFailure(const CheckLocInfo& check_loc_info,
//                              const Node* node,
//                              const std::string& explanation)
//            : CheckFailure(check_loc_info, node_validation_failure_loc_string(node), explanation)
//        {
//        }
//    };
//
// Then, we define the macro NODE_VALIDATION_CHECK as follows:
//
// #define NODE_VALIDATION_CHECK(node, cond, ...) <backslash>
//     NGRAPH_CHECK_HELPER(::ngraph::NodeValidationFailure, (node), (cond), ##__VA_ARGS__)
//
// The macro NODE_VALIDATION_CHECK can now be called on any condition, with a Node* pointer
// supplied to generate an informative error message via node_validation_failure_loc_string().
//
// Take care to fully qualify the exception class name in the macro body.
//
// The "..." may be filled with expressions of any type that has an "operator<<" overload for\
// insertion into std::ostream.
//
// TODO(amprocte): refactor NGRAPH_CHECK_HELPER so we don't have to introduce a locally-scoped
// variable (ss___) and risk shadowing.
//
#define NGRAPH_CHECK_HELPER(exc_class, ctx, check, ...)                                            \
    do                                                                                             \
    {                                                                                              \
        if (!(check))                                                                              \
        {                                                                                          \
            ::std::stringstream ss___;                                                             \
            ::ngraph::write_all_to_stream(ss___, ##__VA_ARGS__);                                   \
            throw exc_class(                                                                       \
                (::ngraph::CheckLocInfo{__FILE__, __LINE__, #check}), (ctx), ss___.str());         \
        }                                                                                          \
    } while (0)

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ngraph::CheckFailure if `cond` is false.
#define NGRAPH_CHECK(cond, ...)                                                                    \
    NGRAPH_CHECK_HELPER(::ngraph::CheckFailure, "", (cond), ##__VA_ARGS__)
