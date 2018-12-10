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

#include <plaidml/plaidml++.h>

#include "ngraph/log.hpp"
#include "ngraph/runtime/plaidml/plaidml_logger.hpp"

namespace
{
    void logger(void* debug, vai_log_severity severity, const char* message)
    {
        switch (severity)
        {
        case VAI_LOG_SEVERITY_VERBOSE:
        case VAI_LOG_SEVERITY_TRACE:
        case VAI_LOG_SEVERITY_DEBUG:
            if (debug)
            {
                PLAIDML_DEBUG << message;
            }
            return;
        case VAI_LOG_SEVERITY_INFO:
            // We treat PlaidML info-level logs as nGraph debug-level logs, since we expect that
            // most nGraph users think of PlaidML details as debugging information.
            if (debug)
            {
                PLAIDML_DEBUG << message;
            }
            return;
        case VAI_LOG_SEVERITY_WARNING: NGRAPH_WARN << message; return;
        case VAI_LOG_SEVERITY_ERROR:
        default: NGRAPH_ERR << message; return;
        }
    }
}

void ngraph::runtime::plaidml::configure_plaidml_logger(bool debug)
{
    vai_set_logger(&logger, reinterpret_cast<void*>(debug ? 1 : 0));
}
