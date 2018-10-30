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

#include "ngraph/log.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            void configure_plaidml_logger(bool debug);

// N.B. This is an unconditional write to the debug log, used when PlaidML debugging is enabled.
#define PLAIDML_DEBUG                                                                              \
    ngraph::LogHelper(ngraph::LOG_TYPE::_LOG_TYPE_DEBUG,                                           \
                      ngraph::get_file_name(__FILE__),                                             \
                      __LINE__,                                                                    \
                      ngraph::default_logger_handler_func)                                         \
        .stream()
        }
    }
}
