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

#include <unordered_map>

namespace ngraph
{
    namespace pass
    {
        class PassConfig;
    }
}

class ngraph::pass::PassConfig
{
public:
    PassConfig();
    PassConfig(const PassConfig& pass_config) { m_enables = pass_config.m_enables; }
    const std::unordered_map<std::string, bool>& get_enables() { return m_enables; }
    void set_pass_enable(std::string name, bool enable);
    bool get_pass_enable(std::string name);

private:
    std::unordered_map<std::string, bool> m_enables;
};
