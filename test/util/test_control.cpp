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

#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "ngraph/log.hpp"
#include "ngraph/util.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

string ngraph::prepend_disabled(const string& test_case_name,
                                const string& test_name,
                                const string& manifest)
{
    static unordered_map<string, unordered_set<string>> s_blacklists;
    cout << __FILE__ << " " << __LINE__ << " " << test_case_name << endl;
    cout << __FILE__ << " " << __LINE__ << " " << test_name << endl;
    cout << __FILE__ << " " << __LINE__ << " " << manifest << endl;
    string rc = test_name;
    cout << __FILE__ << " " << __LINE__ << endl;
    unordered_set<string>& blacklist = s_blacklists[test_case_name];
    cout << __FILE__ << " " << __LINE__ << endl;
    if (blacklist.empty() && !manifest.empty())
    {
        cout << __FILE__ << " " << __LINE__ << endl;
        ifstream f(manifest);
        cout << __FILE__ << " " << __LINE__ << endl;
        string line;
        while (getline(f, line))
        {
            cout << __FILE__ << " " << __LINE__ << " " << line << endl;
            size_t pound_pos = line.find('#');
            cout << __FILE__ << " " << __LINE__ << " " << pound_pos << endl;
            line = (pound_pos > line.size()) ? line : line.substr(0, pound_pos);
            cout << __FILE__ << " " << __LINE__ << " " << line << endl;
            line = trim(line);
            cout << __FILE__ << " " << __LINE__ << " " << line << endl;
            if (line.size() > 1)
            {
                blacklist.insert(line);
            }
        }
    }
    if (contains(blacklist, test_name))
    {
        rc = "DISABLED_" + test_name;
    }
    return rc;
}
