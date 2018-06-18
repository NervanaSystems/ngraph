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
#include <sstream>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

bool runtime::Backend::register_backend(const string& name, shared_ptr<Backend> backend)
{
    get_backend_map().insert({name, backend});
    return true;
}

unordered_map<string, shared_ptr<runtime::Backend>>& runtime::Backend::get_backend_map()
{
    static unordered_map<string, shared_ptr<Backend>> backend_map;
    return backend_map;
}

runtime::Backend::~Backend()
{
}

shared_ptr<runtime::Backend> runtime::Backend::create_dynamic_backend(string type,
                                                                      const OptionsMap& options)
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
    string name = "lib" + to_lower(type) + "_backend" + ext;
    handle = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle)
    {
        string err = dlerror();
        throw runtime_error("Library open for Backend '" + name + "' failed with error:\n" + err);
    }

    auto create = reinterpret_cast<runtime::Backend* (*)(const std::string&, const OptionsMap&)>(
        dlsym(handle, "create_backend"));
    auto destroy = reinterpret_cast<void (*)(runtime::Backend*)>(dlsym(handle, "destroy_backend"));
    if (!create)
    {
        throw runtime_error("Failed to find create_backend function in library '" + name + "'");
    }
    if (!destroy)
    {
        throw runtime_error("Failed to find destroy_backend function in library '" + name + "'");
    }

    Backend* pBackend = create(type, options);
    return shared_ptr<Backend>(pBackend, [destroy](Backend* be) { destroy(be); });
}

shared_ptr<runtime::Backend> runtime::Backend::create(const string& type, const OptionsMap& options)
{
    auto it = get_backend_map().find(type);
    if (it == get_backend_map().end())
    {
        return create_dynamic_backend(type, options);
    }
    it->second->setConfiguration(options);
    return it->second;
}

vector<string> runtime::Backend::get_registered_devices()
{
    vector<string> rc;
    for (const auto& p : get_backend_map())
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
