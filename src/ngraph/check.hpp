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
    static inline std::ostream& write_all_to_stream(std::ostream& str, const T& arg, TS... args)
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
               << ":" << check_loc_info.line << ":" << std::endl;
            ss << context_info << ":" << std::endl;
            ss << explanation << std::endl;
            return ss.str();
        }
    };
}

// TODO(amprocte): refactor so we don't have to introduce a locally-scoped variable and risk
// shadowing here.
#define NGRAPH_CHECK(exc_class, ctx, check, ...)                                                   \
    do                                                                                             \
    {                                                                                              \
        if (!(check))                                                                              \
        {                                                                                          \
            ::std::stringstream ss___;                                                             \
            ::ngraph::write_all_to_stream(ss___, __VA_ARGS__);                                     \
            throw exc_class(                                                                       \
                (::ngraph::CheckLocInfo{__FILE__, __LINE__, #check}), (ctx), ss___.str());         \
        }                                                                                          \
    } while (0)
