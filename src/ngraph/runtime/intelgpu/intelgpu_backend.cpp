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
#include <CPP/layout.hpp>
#include <CPP/network.hpp>
#include <CPP/scale.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"

using namespace std;
using namespace ngraph;

void do_eltwise_operation(cldnn::topology& topology,
                          const shared_ptr<Node>& op,
                          cldnn::eltwise_mode mode)
{
    if (op->get_output_size() != 1 || op->get_input_size() != 2)
    {
        ostringstream os;
        os << "eltwise operation \"" << op->description() << "\" input and output sizes mismatch.";
        throw std::invalid_argument(os.str());
    }

    std::vector<cldnn::primitive_id> op_add_inputs;
    for (const descriptor::Input& op_input : op->get_inputs())
    {
        const std::string& element_name = op_input.get_tensor().get_name();
        op_add_inputs.push_back(element_name);
    }

    const std::string& output_name = op->get_outputs().begin()->get_tensor().get_name();

    cldnn::eltwise op_add(output_name, op_add_inputs, mode);
    topology.add(op_add);
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
    FunctionInstance& instance = ocl_networks[func];
    if (instance.ocl_network != nullptr)
    {
        return true;
    }

    cldnn::topology topology;

    for (shared_ptr<Node> op : func->get_ops())
    {
        if ("Parameter" == op->description())
        {
            if (op->get_output_size() != 1 || op->get_input_size() != 0)
            {
                throw ngraph_error("Parameter input and output sizes mismatch.");
            }
            const std::string& element_name = op->get_output_tensor_view()->get_tensor().get_name();
            cldnn::layout element_layout =
                IntelGPULayout::create_cldnn_layout(op->get_element_type(), op->get_shape());

            cldnn::input_layout op_layout(element_name, element_layout);
            topology.add(op_layout);
        }
        else if ("Add" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::sum);
        }
        else if ("Multiply" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::prod);
        }
        else
        {
            ostringstream os;
            os << "Unsupported operation \"" << op->description() << '\"';
            throw std::invalid_argument(os.str());
        }
    }

    instance.ocl_network = std::make_shared<cldnn::network>(*ocl_engine, topology);

    return true;
}

bool runtime::intelgpu::IntelGPUBackend::call(
    shared_ptr<Function> func,
    const vector<shared_ptr<runtime::TensorView>>& outputs,
    const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    throw runtime_error("IntelGPUBackend::call: Not implemented yet");
}
