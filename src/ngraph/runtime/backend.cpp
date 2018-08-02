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

#ifdef WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <sstream>

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

#ifdef WIN32
#define OPEN_LIBRARY(a, b) LoadLibrary(a)
#define CLOSE_LIBRARY(a) FreeLibrary(a)
#define DLSYM(a, b) GetProcAddress(a, b)
#else
// #define OPEN_LIBRARY(a, b) dlopen(a, b)
#define CLOSE_LIBRARY(a) dlclose(a)
#define DLSYM(a, b) dlsym(a, b)
#endif

runtime::Backend::~Backend()
{
}

// This doodad finds the full path of the containing shared library
static string find_my_file()
{
#ifdef WIN32
    return ".";
#else
    Dl_info dl_info;
    dladdr(reinterpret_cast<void*>(find_my_file), &dl_info);
    return dl_info.dli_fname;
#endif
}

DL_HANDLE runtime::Backend::open_shared_library(string type)
{
    string ext = SHARED_LIB_EXT;

    DL_HANDLE handle;

    // strip off attributes, IE:CPU becomes IE
    auto colon = type.find(":");
    if (colon != type.npos)
    {
        type = type.substr(0, colon);
    }

    string library_name = "lib" + to_lower(type) + "_backend" + string(SHARED_LIB_EXT);
    string my_directory = file_util::get_directory(find_my_file());
    string library_path = file_util::path_join(my_directory, library_name);
#ifdef WIN32
    handle = LoadLibrary(library_path.c_str());
#else
    handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif

    return handle;
}

shared_ptr<runtime::Backend> runtime::Backend::create(const string& type)
{
    shared_ptr<runtime::Backend> rc;
    DL_HANDLE handle = open_shared_library(type);
    if (!handle)
    {
        throw runtime_error("Backend '" + type + "' not found");
    }
    else
    {
        function<const char*()> get_ngraph_version_string =
            reinterpret_cast<const char* (*)()>(DLSYM(handle, "get_ngraph_version_string"));
        if (!get_ngraph_version_string)
        {
            CLOSE_LIBRARY(handle);
            throw runtime_error("Backend '" + type +
                                "' does not implement get_ngraph_version_string");
        }

        function<runtime::Backend*(const char*)> new_backend =
            reinterpret_cast<runtime::Backend* (*)(const char*)>(DLSYM(handle, "new_backend"));
        if (!new_backend)
        {
            CLOSE_LIBRARY(handle);
            throw runtime_error("Backend '" + type + "' does not implement new_backend");
        }

        function<void(runtime::Backend*)> delete_backend =
            reinterpret_cast<void (*)(runtime::Backend*)>(DLSYM(handle, "delete_backend"));
        if (!delete_backend)
        {
            CLOSE_LIBRARY(handle);
            throw runtime_error("Backend '" + type + "' does not implement delete_backend");
        }

        runtime::Backend* backend = new_backend(type.c_str());
        rc = shared_ptr<runtime::Backend>(backend, [=](runtime::Backend* b) {
            delete_backend(b);
            // CLOSE_LIBRARY(handle);
        });
    }
    return rc;
}

map<string, string> runtime::Backend::get_registered_device_map()
{
    map<string, string> rc;
    string my_directory = file_util::get_directory(find_my_file());
    vector<string> backend_list;

    auto f = [&](const string& file, bool is_dir) {
        string name = file_util::get_file_name(file);
        string backend_name;
        if (is_backend_name(name, backend_name))
        {
            DL_HANDLE handle;
#ifdef WIN32
            handle = LoadLibrary(file.c_str());
#else
            handle = dlopen(file.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
            if (handle)
            {
                if (DLSYM(handle, "new_backend") && DLSYM(handle, "delete_backend"))
                {
                    function<const char*()> get_ngraph_version_string =
                        reinterpret_cast<const char* (*)()>(
                            DLSYM(handle, "get_ngraph_version_string"));
                    if (get_ngraph_version_string &&
                        get_ngraph_version_string() == string(NGRAPH_VERSION))
                    {
                        rc.insert({to_upper(backend_name), file});
                    }
                }

                CLOSE_LIBRARY(handle);
            }
        }
    };
    file_util::iterate_files(my_directory, f, false, true);
    return rc;
}

vector<string> runtime::Backend::get_registered_devices()
{
    map<string, string> m = get_registered_device_map();
    vector<string> rc;
    for (const pair<string, string>& p : m)
    {
        rc.push_back(p.first);
    }
    return rc;
}

void runtime::Backend::remove_compiled_function(shared_ptr<Function> func)
{
}

vector<ngraph::runtime::PerformanceCounter>
    runtime::Backend::get_performance_data(shared_ptr<Function> func) const
{
    return vector<PerformanceCounter>();
}

void runtime::Backend::validate_call(shared_ptr<const Function> function,
                                     const vector<shared_ptr<runtime::TensorView>>& outputs,
                                     const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    const op::ParameterVector& input_parameters = function->get_parameters();
    if (input_parameters.size() != inputs.size())
    {
        stringstream ss;
        ss << "Call input count " << inputs.size() << " does not match Function's Parameter count "
           << input_parameters.size();
        throw runtime_error(ss.str());
    }
    if (function->get_output_size() != outputs.size())
    {
        stringstream ss;
        ss << "Call output count " << outputs.size() << " does not match Function's Result count "
           << function->get_output_size();
        throw runtime_error(ss.str());
    }

    for (size_t i = 0; i < input_parameters.size(); i++)
    {
        if (input_parameters[i]->get_element_type() != inputs[i]->get_tensor().get_element_type())
        {
            stringstream ss;
            ss << "Input " << i << " type '" << inputs[i]->get_tensor().get_element_type()
               << "' does not match Parameter type '" << input_parameters[i]->get_element_type()
               << "'";
            throw runtime_error(ss.str());
        }
        if (input_parameters[i]->get_shape() != inputs[i]->get_shape())
        {
            stringstream ss;
            ss << "Input " << i << " shape {" << join(inputs[i]->get_shape())
               << "} does not match Parameter shape {" << join(input_parameters[i]->get_shape())
               << "}";
            throw runtime_error(ss.str());
        }
    }

    for (size_t i = 0; i < function->get_output_size(); i++)
    {
        if (function->get_output_element_type(i) != outputs[i]->get_tensor().get_element_type())
        {
            stringstream ss;
            ss << "Output " << i << " type '" << outputs[i]->get_tensor().get_element_type()
               << "' does not match Parameter type '" << function->get_output_element_type(i)
               << "'";
            throw runtime_error(ss.str());
        }
        if (function->get_output_shape(i) != outputs[i]->get_shape())
        {
            stringstream ss;
            ss << "Output " << i << " shape {" << join(outputs[i]->get_shape())
               << "} does not match Parameter shape {" << join(function->get_output_shape(i))
               << "}";
            throw runtime_error(ss.str());
        }
    }
}

bool runtime::Backend::is_backend_name(const string& file, string& backend_name)
{
    string name = file_util::get_file_name(file);
    string ext = SHARED_LIB_EXT;
    bool rc = false;
    if (!name.compare(0, 3, "lib"))
    {
        if (!name.compare(name.size() - ext.size(), ext.size(), ext))
        {
            auto pos = name.find("_backend");
            if (pos != name.npos)
            {
                backend_name = name.substr(3, pos - 3);
                rc = true;
            }
        }
    }
    return rc;
}
