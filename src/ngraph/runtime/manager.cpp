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

using namespace ngraph::runtime;

static std::mutex load_plugins_mutex;
static std::mutex close_plugins_mutex;

bool Manager::m_is_factory_map_initialized = false;
std::vector<void*> Manager::m_plugin_handles = {};

void Manager::load_plugins(const std::string& runtime_plugin_libs)
{
    std::lock_guard<std::mutex> lock(load_plugins_mutex);

    if (Manager::m_is_factory_map_initialized)
    {
        return;
    }

    std::vector<std::string> plugin_paths = ngraph::split(runtime_plugin_libs, ':', false);
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
                    Manager::m_plugin_handles.push_back(plugin_handle);
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

    Manager::m_is_factory_map_initialized = true;
}

// TODO: Should call this function after plugin is not needed anymore.
void Manager::close_plugins()
{
    std::lock_guard<std::mutex> lock(close_plugins_mutex);

    for (auto plugin_handle : Manager::m_plugin_handles)
    {
        dlclose(plugin_handle);
    }
    Manager::m_plugin_handles.clear();
}

Manager::FactoryMap& Manager::get_factory_map()
{
    // Stores Manager Factories
    static FactoryMap factory_map;
    return factory_map;
}

std::shared_ptr<Manager> Manager::get(const std::string& name)
{
    Manager::load_plugins(RUNTIME_PLUGIN_LIBS);

    auto iter = get_factory_map().find(name);

    if (iter == get_factory_map().end())
    {
        throw ngraph_error("No nGraph runtime with name '" + name + "' has been registered.");
    }

    Factory& f = iter->second;
    return f(name);
}

Manager::Factory Manager::register_factory(const std::string& name, Factory factory)
{
    get_factory_map()[name] = factory;
    return factory;
}
