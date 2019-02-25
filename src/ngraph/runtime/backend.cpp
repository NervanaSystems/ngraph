//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <sstream>

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::Backend::~Backend()
{
}

unique_ptr<runtime::Backend> runtime::Backend::create(const string& type)
{
    return BackendManager::create_backend(type);
}

vector<string> runtime::Backend::get_registered_devices()
{
    return BackendManager::get_registered_backends();
}

bool runtime::Backend::is_supported(const Node& node) const
{
    // The default behavior is that a backend does not support any ops. If this is not the case
    // then override this method and enhance.
    return false;
}

runtime::Executable::Executable()
{
}

runtime::Executable::~Executable()
{
}

bool runtime::Executable::call_with_validate(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                             const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    validate(outputs, inputs);
    return call(outputs, inputs);
}

void runtime::Executable::validate(const vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                   const vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    const ParameterVector& parameters = get_parameters();
    const ResultVector& results = get_results();
    if (parameters.size() != inputs.size())
    {
        stringstream ss;
        ss << "Call input count " << inputs.size() << " does not match Function's Parameter count "
           << parameters.size();
        throw runtime_error(ss.str());
    }
    if (results.size() != outputs.size())
    {
        stringstream ss;
        ss << "Call output count " << outputs.size() << " does not match Function's Result count "
           << results.size();
        throw runtime_error(ss.str());
    }

    for (size_t i = 0; i < parameters.size(); i++)
    {
        if (parameters[i]->get_element_type() != inputs[i]->get_element_type())
        {
            stringstream ss;
            ss << "Input " << i << " type '" << inputs[i]->get_element_type()
               << "' does not match Parameter type '" << parameters[i]->get_element_type() << "'";
            throw runtime_error(ss.str());
        }
        if (parameters[i]->get_shape() != inputs[i]->get_shape())
        {
            stringstream ss;
            ss << "Input " << i << " shape {" << join(inputs[i]->get_shape())
               << "} does not match Parameter shape {" << join(parameters[i]->get_shape()) << "}";
            throw runtime_error(ss.str());
        }
    }

    for (size_t i = 0; i < results.size(); i++)
    {
        if (results[i]->get_element_type() != outputs[i]->get_element_type())
        {
            stringstream ss;
            ss << "Output " << i << " type '" << outputs[i]->get_element_type()
               << "' does not match Result type '" << results[i]->get_element_type() << "'";
            throw runtime_error(ss.str());
        }
        if (results[i]->get_shape() != outputs[i]->get_shape())
        {
            stringstream ss;
            ss << "Output " << i << " shape {" << join(outputs[i]->get_shape())
               << "} does not match Result shape {" << join(results[i]->get_shape()) << "}";
            throw runtime_error(ss.str());
        }
    }
}

const ngraph::ParameterVector& runtime::Executable::get_parameters() const
{
    return m_parameters;
}

const ngraph::ResultVector& runtime::Executable::get_results() const
{
    return m_results;
}

void runtime::Executable::set_parameters_and_results(const Function& func)
{
    m_parameters = func.get_parameters();
    m_results = func.get_results();
}

vector<runtime::PerformanceCounter> runtime::Executable::get_performance_data() const
{
    return vector<PerformanceCounter>();
}

bool runtime::Backend::is_supported_property(const Property prop) const
{
    return false;
}

void runtime::Backend::remove_compiled_function(std::shared_ptr<Executable> exec)
{
}
