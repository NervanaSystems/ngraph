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

#include <sstream>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::shared_ptr<runtime::Backend> runtime::Backend::create(const std::string& type)
{
    std::shared_ptr<Manager> manager = runtime::Manager::get(type);
    return manager->allocate_backend();
}

vector<string> runtime::Backend::get_registered_devices()
{
    vector<string> rc;
    for (const pair<string, runtime::Manager::Factory>& p : runtime::Manager::get_factory_map())
    {
        rc.push_back(p.first);
    }
    return rc;
}

vector<size_t> runtime::Backend::get_subdevices(const string& type)
{
    std::shared_ptr<Manager> manager = runtime::Manager::get(type);
    return manager->get_subdevices();
}

void runtime::Backend::remove_compiled_function(std::shared_ptr<Function> func)
{
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
