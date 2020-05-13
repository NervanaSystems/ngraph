//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/runtime/ie/ie_executable.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "ngraph/runtime/ie/ie_tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

runtime::ie::IE_Executable::IE_Executable(shared_ptr<Function> func, string device)
    : m_device{device}
{
    const auto& opset = get_opset1();
    pass::Manager passes;
    passes.register_pass<pass::Opset1Upgrade>();
    passes.run_passes(func);

    for (const auto& node : func->get_ops())
    {
        if (!opset.contains_op_type(node.get()))
        {
            if (node->get_type_info() == op::GetOutputElement::type_info)
            {
                // IE currently can handle GetOutuputElement op;
                continue;
            }
            else
            {
                cout << "UNSUPPORTED OP DETECTED: " << node->get_type_info().name << endl;
                THROW_IE_EXCEPTION << "Detected op not belonging to opset1!";
            }
        }
    }

#ifdef NGRAPH_DEBUG_ENABLE
    cout << "Nodes in test: ";
    for (const auto& node : func->get_ops())
    {
        cout << node << endl;
    }
    cout << endl;
#endif

    m_network = InferenceEngine::CNNNetwork(func);
    set_parameters_and_results(*func);

    InferenceEngine::Core ie;
    //  Load model to the plugin (BACKEND_NAME)
    InferenceEngine::ExecutableNetwork exe_network = ie.LoadNetwork(m_network, m_device);
    //  Create infer request
    m_infer_req = exe_network.CreateInferRequest();
}

bool runtime::ie::IE_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                      const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // Prepare input blobs
    InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();
    if (input_info.size() != inputs.size())
    {
        THROW_IE_EXCEPTION << "Function inputs number differ from number of given inputs";
    }

    size_t i = 0;
    for (const auto& it : input_info)
    {
        string input_name = it.first;
        InferenceEngine::Blob::Ptr input_blob = m_infer_req.GetBlob(input_name);
        auto input_buffer = input_blob->buffer().as<void*>();
        size_t input_data_size = input_blob->byteSize();
        inputs[i]->read(input_buffer, input_data_size);
        i++;
    }

    // Infer Request
    m_infer_req.Infer();

    // Prepare output blobs
    InferenceEngine::OutputsDataMap output_info = m_network.getOutputsInfo();
    if (output_info.size() != outputs.size())
    {
        THROW_IE_EXCEPTION << "Function outputs number differ from number of given outputs";
    }

    i = 0;
    for (const auto& it : output_info)
    {
        string output_name = it.first;
        InferenceEngine::Blob::Ptr output_blob = m_infer_req.GetBlob(output_name);
        auto output_buffer = output_blob->buffer().as<void*>();
        size_t output_data_size = output_blob->byteSize();
        outputs[i]->write(output_buffer, output_data_size);
        i++;
    }
    return true;
}
