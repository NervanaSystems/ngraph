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

#include <cstring>
#include <sstream>

#include "ngraph/log.hpp"
#include "ngraph/runtime/plaidml/plaidml_config.hpp"
#include "ngraph/runtime/plaidml/plaidml_logger.hpp"

namespace v = vertexai;
namespace vp = vertexai::plaidml;

extern "C" void vai_internal_set_vlog(std::size_t num);

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            namespace
            {
                vp::device get_device(const std::shared_ptr<vertexai::ctx>& ctx,
                                      std::size_t device_idx)
                {
                    auto dev_configs = vp::enumerate_devices(ctx);
                    if (!dev_configs.size())
                    {
                        throw std::runtime_error{"Unable to find any PlaidML devices"};
                    }
                    if (dev_configs.size() <= device_idx)
                    {
                        throw std::runtime_error{"Device index out of range"};
                    }
                    return dev_configs[device_idx].open();
                }

                void list_devices(const std::shared_ptr<vertexai::ctx>& ctx)
                {
                    auto dev_configs = vp::enumerate_devices(ctx);
                    if (!dev_configs.size())
                    {
                        NGRAPH_WARN << "No PlaidML devices found";
                        return;
                    }
                    NGRAPH_INFO << "PlaidML Devices:";
                    for (std::size_t idx = 0; idx < dev_configs.size(); ++idx)
                    {
                        const auto& config = dev_configs[idx];
                        NGRAPH_INFO << "\t" << idx << ": " << config.id() << ": "
                                    << config.description();
                    }
                }
            }
        }
    }
}

ngraph::runtime::plaidml::Config
    ngraph::runtime::plaidml::parse_config_string(const char* configuration_string)
{
    bool err = false;
    bool help = false;
    bool list = false;
    bool debug = false;
    std::size_t device_idx = 0;
    std::string eventlog_config;

#ifdef NGRAPH_DEBUG_ENABLE
    debug = true;
#endif

    // To visualize what's going on here, here's a configuration string fragment:
    //
    //     ,option_name=option_value,
    //      ^          ^^           ^
    // oname_begin     ||           |
    //         oname_end|           |
    //                  oval_begin  |
    //                          oval_end
    //
    // When there is no option value, here's where the pointers go:
    //
    //     ,option_name,
    //      ^          ^
    // oname_begin     |
    //         oname_end
    //        oval_begin
    //          oval_end

    const char* c = configuration_string;
    while (*c && *c != ':')
    {
        ++c;
    }

    // Before the options, we have an optional device index.
    if (*c)
    {
        char* dev_end;
        std::size_t explicit_idx = std::strtoul(c + 1, &dev_end, 10);
        if (dev_end != c + 1)
        {
            device_idx = explicit_idx;
            c = dev_end;
        }
    }

    while (*c)
    {
        // Invariant: c points to the character introducing the current option.

        const char* oname_begin = c + 1;
        // Invariant: oname_begin points to the first character of the option name.

        const char* oname_end = oname_begin;
        while (*oname_end && *oname_end != '=' && *oname_end != ',')
        {
            ++oname_end;
        }
        // Invariant: [oname_begin, oname_end) is the option name.

        const char* oval_begin = oname_end;
        if (*oval_begin == '=')
        {
            ++oval_begin;
        }
        const char* oval_end = oval_begin;
        while (*oval_end && *oval_end != ',')
        {
            ++oval_end;
        }
        // Invariant: [oval_begin, oval_end) is the option value.

        // Re-establish initial invariant, allowing "continue" to resume the loop.
        c = oval_end;

        // Readability definitions
        auto is_opt = [=](const char* opt) {
            auto len = strlen(opt);
            return (oname_end - oname_begin == len) && !strncmp(oname_begin, opt, len);
        };

        auto oval_len = oval_end - oval_begin;
        bool has_oval = oval_begin != oname_end;

        // N.B. oval_len != 0 => has_oval, but there's no other relationship.
        // So to verify that there is a non-zero-length option value, test oval_len
        // To verify that there is no option value, test has_oval

        // Check for verbosity
        if (is_opt("v"))
        {
            if (!oval_len)
            {
                throw std::invalid_argument{"PlaidML verbosity level requires a value"};
            }
            char* val_end;
            std::size_t vlog = std::strtoul(oval_begin, &val_end, 10);
            if (oval_end != val_end)
            {
                throw std::invalid_argument{"Invalid PlaidML verbosity level"};
            }
            debug = true;
            vai_internal_set_vlog(vlog);
            continue;
        }

        // Check for help
        if (is_opt("help"))
        {
            help = true;
            continue;
        }

        // Check for PlaidML debugging
        if (is_opt("debug"))
        {
            debug = true;
            continue;
        }

        // Check for list_devices
        if (is_opt("list_devices"))
        {
            if (has_oval)
            {
                throw std::invalid_argument{"PlaidML list_devices does not take a value"};
            }
            list = true;
            continue;
        }

        // Check for eventlog
        if (is_opt("eventlog"))
        {
            if (!oval_len)
            {
                throw std::invalid_argument{"PlaidML eventlog requires a value"};
            }
            std::ostringstream e;
            e << "{\"@type\": "
                 "\"type.vertex.ai/vertexai.eventing.file.proto.EventLog\", "
                 "\"filename\": \"";
            for (const char* oc = oval_begin; oc < oval_end; ++oc)
            {
                if (!isalnum(*oc))
                {
                    e << '\\';
                }
                e << *oc;
            }
            e << "\"}";
            eventlog_config = e.str();
            continue;
        }

        // Reject unknown options
        err = true;
    }

    constexpr char help_text[] =
        "PlaidML Backend Specification: \""
        "PlaidML[:[device_index][,debug][,help][,list_devices][,"
        "eventlog=<filename>]]\".  For example: \"PlaidML\", \""
        "PlaidML:0,list_devices\"";
    if (err)
    {
        NGRAPH_ERR << help_text;
        throw std::invalid_argument{"Invalid parameter supplied to PlaidML backend"};
    }

    if (help)
    {
        NGRAPH_INFO << help_text;
    }

    // Ensure process-level logging callbacks are in place.
    configure_plaidml_logger(debug);

    // Build the PlaidML configuration.
    Config result;

    result.ctx = std::make_shared<vertexai::ctx>();
    if (eventlog_config.length())
    {
        v::vai_exception::check_and_throw(
            vai_set_eventlog(result.ctx->get_ctx(), eventlog_config.c_str()));
    }
    if (list)
    {
        list_devices(result.ctx);
    }
    result.dev = std::make_shared<vertexai::plaidml::device>(get_device(result.ctx, device_idx));

    result.debug = debug;

    return result;
}
