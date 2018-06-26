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

#include <memory>
#include <string>
#include <unordered_map>

namespace ngraph
{
    namespace runtime
    {
        class BackendCache;
        class Backend;
    }
}

class ngraph::runtime::BackendCache
{
public:
    static std::shared_ptr<Backend> create(const std::string& type);
    static bool register_backend(const std::string& name, std::shared_ptr<Backend>);

private:
    static void* open_shared_library(std::string type);
    static std::unordered_map<std::string, std::shared_ptr<Backend>>& get_backend_map();
    static std::unordered_map<std::string, void*> s_open_backends;
};
