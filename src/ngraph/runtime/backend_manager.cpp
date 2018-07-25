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

#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

unordered_map<string, runtime::new_backend_t>& runtime::BackendManager::get_registry()
{
    static unordered_map<string, new_backend_t> s_registered_backend;
    return s_registered_backend;
}

void runtime::BackendManager::register_backend(const string& name, new_backend_t new_backend)
{
    NGRAPH_INFO << name;
    get_registry()[name] = new_backend;
}

vector<string> runtime::BackendManager::get_registered_backends()
{
    vector<string> rc;

    return rc;
}

shared_ptr<runtime::Backend> runtime::BackendManager::create_backend(const std::string& config)
{
    shared_ptr<runtime::Backend> rc;
    string type = config;

    // strip off attributes, IE:CPU becomes IE
    auto colon = type.find(":");
    if (colon != type.npos)
    {
        type = type.substr(0, colon);
    }

    auto registry = get_registry();
    auto it = registry.find(type);
    if (it == registry.end())
    {
        stringstream ss;
        ss << "Backend '" << type << "' not registered";
        throw runtime_error(ss.str());
    }
    new_backend_t new_backend = it->second;
    rc = new_backend(config);
    return rc;
}
