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
#include <sstream>
#include <string>

#include "ngraph/except.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/util.hpp"

using namespace ngraph::runtime;

bool Manager::m_is_factory_map_initialized = false;
std::shared_ptr<std::vector<void*>> Manager::m_plugin_lib_handles =
    std::make_shared<std::vector<void*>>(std::vector<void*>());

void Manager::load_plugins(const std::string& runtime_plugin_libs)
{
    std::vector<std::string> plugin_lib_paths = ngraph::split(runtime_plugin_libs, ':', false);

    for (auto plugin_lib_path : plugin_lib_paths)
    {
        if (plugin_lib_path.size() > 0)
        {
            void* lib_handle = dlopen(plugin_lib_path.c_str(), RTLD_NOW);
            if (lib_handle)
            {
                Manager::m_plugin_lib_handles->push_back(lib_handle);
            }
            else
            {
                throw ngraph_error("Cannot open library " + plugin_lib_path);
            }
        }
    }
}

// TODO: Should call this function after plugin is not needed anymore.
void Manager::close_plugins()
{
    for (auto lib_handle : *Manager::m_plugin_lib_handles)
    {
        dlclose(lib_handle);
    }
    Manager::m_plugin_lib_handles->clear();
}

Manager::FactoryMap& Manager::get_factory_map()
{
    // Stores Manager Factories
    static FactoryMap factory_map;

    // Try to load runtime plugins
    if (!Manager::m_is_factory_map_initialized)
    {
        Manager::load_plugins(RUNTIME_PLUGIN_LIBS);
        Manager::m_is_factory_map_initialized = true;
    }
    return factory_map;
}

std::shared_ptr<Manager> Manager::get(const std::string& name)
{
    return get_factory_map().at(name)(name);
}

Manager::Factory Manager::register_factory(const std::string& name, Factory factory)
{
    get_factory_map()[name] = factory;
    return factory;
}
