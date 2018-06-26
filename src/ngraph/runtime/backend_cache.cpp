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

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend_cache.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

unordered_map<string, void*> runtime::BackendCache::s_open_backends;

// This doodad finds the full path of the containing shared library
static string find_my_file()
{
    Dl_info dl_info;
    dladdr(reinterpret_cast<void*>(find_my_file), &dl_info);
    return dl_info.dli_fname;
}

// This will be uncommented when we add support for listing all known backends
// static bool is_backend(const string& path)
// {
//     bool rc = false;
//     string name = file_util::get_file_name(path);
//     if (name.find("_backend.") != string::npos)
//     {
//         NGRAPH_INFO << name;
//     }
//     return rc;
// }

void* runtime::BackendCache::open_shared_library(string type)
{
    string ext = SHARED_LIB_EXT;
    string ver = LIBRARY_VERSION;

    void* handle = nullptr;

    // strip off attributes, IE:CPU becomes IE
    auto colon = type.find(":");
    if (colon != type.npos)
    {
        type = type.substr(0, colon);
    }
    string lib_name = "lib" + to_lower(type) + "_backend" + ext;
    string my_directory = file_util::get_directory(find_my_file());
    string full_path = file_util::path_join(my_directory, lib_name);
    handle = dlopen(full_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle)
    {
        function<void()> create_backend =
            reinterpret_cast<void (*)()>(dlsym(handle, "create_backend"));
        if (create_backend)
        {
            create_backend();
        }
        else
        {
            dlclose(handle);
            throw runtime_error("Failed to find create_backend function in library '" + lib_name +
                                "'");
        }
        s_open_backends.insert({lib_name, handle});
    }
    else
    {
        string err = dlerror();
        throw runtime_error("Library open for Backend '" + lib_name + "' failed with error:\n" +
                            err);
    }
    return handle;
}

bool runtime::BackendCache::register_backend(const string& name, shared_ptr<Backend> backend)
{
    get_backend_map().insert({name, backend});
    return true;
}

shared_ptr<runtime::Backend> runtime::BackendCache::create(const string& type)
{
    auto it = get_backend_map().find(type);
    if (it == get_backend_map().end())
    {
        open_shared_library(type);
        it = get_backend_map().find(type);
        if (it == get_backend_map().end())
        {
            throw runtime_error("Backend '" + type + "' not found in registered backends.");
        }
    }
    return it->second;
}

unordered_map<string, shared_ptr<runtime::Backend>>& runtime::BackendCache::get_backend_map()
{
    static unordered_map<string, shared_ptr<Backend>> backend_map;
    return backend_map;
}
