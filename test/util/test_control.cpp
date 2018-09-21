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

#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "ngraph/log.hpp"
#include "ngraph/util.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static unordered_map<string, unordered_set<string>> s_blacklists;

string ngraph::prepend_disabled(const string& test_case_name,
                                const string& test_name,
                                const string& manifest)
{
    string rc = test_name;
    unordered_set<string>& blacklist = s_blacklists[test_case_name];
    if (blacklist.empty() && !manifest.empty())
    {
        ifstream f(manifest);
        string line;
        while (getline(f, line))
        {
            size_t pound_pos = line.find('#');
            line = (pound_pos > line.size()) ? line : line.substr(0, pound_pos);
            line = trim(line);
            if (line.size() > 1)
            {
                blacklist.insert(line);
            }
        }
    }
    if (blacklist.find(test_name) != blacklist.end())
    {
        rc = "DISABLED_" + test_name;
    }
    return rc;
}
