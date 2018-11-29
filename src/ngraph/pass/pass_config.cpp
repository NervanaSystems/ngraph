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

#include "ngraph/pass/pass_config.hpp"
#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

ngraph::pass::PassConfig::PassConfig()
{
    /**
    * Parses the semi-colon separated environment string passed through NGRAPH_PASS_ENABLES
    * and returns the pass names and whether they should be enabled or disabled in the
    * provided unordered_map. Implementation of pass selection is up to the backend
    * E.g., NGRAPH_PASS_ENABLES="CoreFusion:0;LikeReplacement:1;CPUCollapseDims" would 
    *       set disables on CoreFusion and enables on LikeReplacement and CPUCollapseDims
    **/
    const char* env_str = std::getenv("NGRAPH_PASS_ENABLES");
    if (env_str)
    {
        std::stringstream ss;
        ss << env_str;
        while (ss.good())
        {
            std::string substr;
            std::getline(ss, substr, ';');
            auto split_str = split(substr, ':', false);
            switch (split_str.size())
            {
            case 1: m_enables.emplace(split_str[0], true); break;
            case 2: m_enables.emplace(split_str[0], parse_string<bool>(split_str[1])); break;
            default: throw ngraph_error("Unexpected string in get_pass_enables: " + substr);
            }
        }
    }
}

void ngraph::pass::PassConfig::set_pass_enable(std::string name, bool enable)
{
    m_enables[name] = enable;
}

bool ngraph::pass::PassConfig::get_pass_enable(std::string name)
{
    if (m_enables.find(name) == m_enables.end())
    {
        return false;
    }
    return m_enables[name];
}
