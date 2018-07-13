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

#include <CPP/eltwise.hpp>
#include <CPP/input_layout.hpp>
#include <CPP/network.hpp>
#include <CPP/topology.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"

using namespace std;
using namespace ngraph;

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::intelgpu::IntelGPUBackend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

runtime::intelgpu::IntelGPUBackend::IntelGPUBackend()
{
    ocl_engine = std::make_shared<cldnn::engine>();
}

shared_ptr<runtime::TensorView>
    runtime::intelgpu::IntelGPUBackend::create_tensor(const element::Type& element_type,
                                                      const Shape& shape)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(element_type, shape, *ocl_engine);
}

shared_ptr<runtime::TensorView> runtime::intelgpu::IntelGPUBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(
        element_type, shape, *ocl_engine, memory_pointer);
}

bool runtime::intelgpu::IntelGPUBackend::compile(shared_ptr<Function> func)
{
    throw runtime_error("IntelGPUBackend::compile: Not implemented yet");
}

bool runtime::intelgpu::IntelGPUBackend::call(
    shared_ptr<Function> func,
    const vector<shared_ptr<runtime::TensorView>>& outputs,
    const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    validate_call(func, outputs, inputs);

    FunctionInstance& instance = ocl_networks[func];
    if (instance.ocl_network == nullptr)
    {
        if (!compile(func))
        {
            return false;
        }
    }

    std::shared_ptr<cldnn::network> network = instance.ocl_network;

    // Process input parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_parameters and inputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < inputs.size(); i++)
    {
        shared_ptr<runtime::intelgpu::IntelGPUTensorView> tv =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(inputs[i]);
        const op::ParameterVector& input_params = func->get_parameters();
        network->set_input_data(input_params[i]->get_output_tensor().get_name(),
                                *tv->get_data_ptr());
    }

    // Execute network
    std::map<cldnn::primitive_id, cldnn::network_output> result = network->execute();

    // Process output parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_results and outputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < func->get_output_size(); i++)
    {
        shared_ptr<runtime::intelgpu::IntelGPUTensorView> ngraph_res =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(outputs[i]);
        const std::string& tensor_name = func->get_output_op(i)->get_output_tensor().get_name();

        auto result_memory = result.at(tensor_name).get_memory().pointer<char>();
        ngraph_res->write(result_memory.data(), 0, result_memory.size());
    }

    return true;
}
