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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef WIN32
#include <windows.h>
#define DL_HANDLE HMODULE
#else
#define DL_HANDLE void*
#endif

namespace ngraph
{
    namespace runtime
    {
        class Backend;
        class BackendManager;

        using new_backend_t = std::function<Backend*(const char* config)>;
    }
}

class ngraph::runtime::BackendManager
{
    friend class Backend;

public:
    /// @brief Used by build-in backends to register their name and constructor.
    ///    This function is not used if the backend is build as a shared library.
    /// @param name The name of the registering backend in UPPER CASE.
    /// @param backend_constructor A function of type new_backend_t which will be called to
    ////     construct an instance of the registered backend.
    static void register_backend(const std::string& name, new_backend_t backend_constructor);

    /// @brief Query the list of registered devices
    /// @returns A vector of all registered devices.
    static std::vector<std::string> get_registered_backends();

private:
    static std::shared_ptr<runtime::Backend> create_backend(const std::string& type);
    static std::unordered_map<std::string, new_backend_t>& get_registry();

    static std::unordered_map<std::string, new_backend_t> s_registered_backend;

    static DL_HANDLE open_shared_library(std::string type);
    static std::map<std::string, std::string> get_registered_device_map();
    static bool is_backend_name(const std::string& file, std::string& backend_name);
};
