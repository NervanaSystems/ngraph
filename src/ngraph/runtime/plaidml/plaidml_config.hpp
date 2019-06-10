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

#include <plaidml/plaidml++.h>

#include <memory>
#include <string>

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            struct Config;

            Config parse_config_string(const std::string& configuration_string);
        }
    }
}

struct ngraph::runtime::plaidml::Config
{
    std::shared_ptr<vertexai::ctx> ctx;
    std::shared_ptr<vertexai::plaidml::device> dev;
    bool debug;
    std::string graphviz;
};
