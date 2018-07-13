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

#include <CPP/concatenation.hpp>
#include <CPP/eltwise.hpp>
#include <CPP/input_layout.hpp>
#include <CPP/layout.hpp>
#include <CPP/network.hpp>
#include <CPP/reorder.hpp>
#include <CPP/scale.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"

using namespace std;
using namespace ngraph;

void arguments_check(const shared_ptr<Node>& op, size_t input, size_t output)
{
    if (op->get_input_size() != input || op->get_output_size() != output)
    {
        ostringstream os;
        os << "Operation \"" << op->description() << "\" input and output sizes mismatch.\n"
           << "Expected input size=" << op->get_input_size() << ", provided=" << input << "\n"
           << "Expected output size=" << op->get_output_size() << ", provided=" << output;
        throw std::invalid_argument(os.str());
    }
}

void do_eltwise_operation(cldnn::topology& topology,
                          const shared_ptr<Node>& op,
                          cldnn::eltwise_mode mode)
{
    arguments_check(op, 2, 1);

    std::vector<cldnn::primitive_id> op_add_inputs;
    for (const descriptor::Input& op_input : op->get_inputs())
    {
        const std::string& element_name = op_input.get_tensor().get_name();
        op_add_inputs.push_back(element_name);
    }

    const std::string& output_name = op->get_outputs().begin()->get_tensor().get_name();

    const cldnn::eltwise op_add(output_name, op_add_inputs, mode);
    topology.add(op_add);
}

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
            arguments_check(op, 0, 1);

            const std::string& element_name = op->get_output_tensor_view()->get_tensor().get_name();
            const cldnn::layout element_layout =
                IntelGPULayout::create_cldnn_layout(op->get_element_type(), op->get_shape());

            const cldnn::input_layout op_layout(element_name, element_layout);
            topology.add(op_layout);
        }
        else if ("Result" == op->description())
        {
            arguments_check(op, 1, 1);

            const descriptor::Tensor& input_tensor = op->get_inputs().begin()->get_tensor();
            const descriptor::Tensor& output_tensor = op->get_outputs().begin()->get_tensor();
            const std::string& input_name = input_tensor.get_name();
            const std::string& output_name = output_tensor.get_name();
            const cldnn::layout input_layout = IntelGPULayout::create_cldnn_layout(
                input_tensor.get_element_type(), op->get_inputs().begin()->get_shape());

            const cldnn::reorder op_reorder(output_name, input_name, input_layout);
            topology.add(op_reorder);
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
