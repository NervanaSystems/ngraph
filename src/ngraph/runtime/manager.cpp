// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#include "ngraph/except.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

static mutex load_plugins_mutex;
static mutex close_plugins_mutex;

bool runtime::Manager::m_is_factory_map_initialized = false;
vector<void*> runtime::Manager::m_plugin_handles = {};

void runtime::Manager::load_plugins(const string& runtime_plugin_libs)
{
    lock_guard<mutex> lock(load_plugins_mutex);

    if (m_is_factory_map_initialized)
    {
        return;
    }

    vector<string> plugin_paths = ngraph::split(runtime_plugin_libs, ':', false);
    for (auto plugin_path : plugin_paths)
    {
        if (plugin_path.size() > 0)
        {
            void* plugin_handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (plugin_handle)
            {
                void (*register_plugin)() =
                    reinterpret_cast<void (*)()>(dlsym(plugin_handle, "register_plugin"));
                if (register_plugin != NULL)
                {
                    register_plugin();
                    m_plugin_handles.push_back(plugin_handle);
                }
                else
                {
                    throw ngraph_error("register_plugin() not found in " + plugin_path);
                }
            }
            else
            {
                throw ngraph_error("Cannot open library " + plugin_path);
            }
        }
    }

    m_is_factory_map_initialized = true;
}

// TODO: Should call this function after plugin is not needed anymore.
void runtime::Manager::close_plugins()
{
    lock_guard<mutex> lock(close_plugins_mutex);

    for (auto plugin_handle : m_plugin_handles)
    {
        dlclose(plugin_handle);
    }
    m_plugin_handles.clear();
}

runtime::Manager::FactoryMap& runtime::Manager::get_factory_map()
{
    // Stores Manager Factories
    static FactoryMap factory_map;
    return factory_map;
}

shared_ptr<runtime::Manager> runtime::Manager::get(const string& name)
{
    load_plugins(RUNTIME_PLUGIN_LIBS);

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
