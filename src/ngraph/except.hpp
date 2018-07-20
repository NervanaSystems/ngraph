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
#include <stdexcept>

namespace ngraph
{
    /// Base error for ngraph runtime errors.
    struct ngraph_error : std::runtime_error
    {
        explicit ngraph_error(const std::string& what_arg)
            : std::runtime_error(what_arg)
        {
        }

        explicit ngraph_error(const char* what_arg)
            : std::runtime_error(what_arg)
        {
        }

        explicit ngraph_error(const std::stringstream& what_arg)
            : std::runtime_error(what_arg.str())
        {
        }
    };
}
