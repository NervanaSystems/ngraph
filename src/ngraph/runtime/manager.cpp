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

#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

// Put all runtime plugin here. Plugins are statically linked, and a known function need to be
// called to register the backend.
#ifdef NGRAPH_NNP_ENABLE
bool REGISTER_NNP_RUNTIME();
static bool nnp_runtime_initialized = REGISTER_NNP_RUNTIME();
#endif

runtime::Manager::FactoryMap& runtime::Manager::get_factory_map()
{
    // Stores Manager Factories
    static FactoryMap factory_map;
    return factory_map;
}

shared_ptr<runtime::Manager> runtime::Manager::get(const string& name)
{
    auto iter = get_factory_map().find(name);

    if (iter == get_factory_map().end())
    {
        throw ngraph_error("No nGraph runtime with name '" + name + "' has been registered.");
    }

    Factory& f = iter->second;
    return f(name);
}

runtime::Manager::Factory runtime::Manager::register_factory(const string& name, Factory factory)
{
    get_factory_map()[name] = factory;
    return factory;
}
