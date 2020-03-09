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

using namespace std;
using namespace ngraph;

runtime::ie::IE_Executable::IE_Executable(shared_ptr<Function> func, string device)
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
    m_device = device == "IE" ? "CPU" : device;
    set_parameters_and_results(*func);
}

bool runtime::ie::IE_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                      const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    InferenceEngine::Core ie;

    //  Loading model to the plugin (BACKEND_NAME)
    InferenceEngine::ExecutableNetwork exe_network = ie.LoadNetwork(m_network, m_device);
    //  Create infer request
    InferenceEngine::InferRequest infer_request = exe_network.CreateInferRequest();
    //  Prepare input and output blobs
    InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();

    if (input_info.size() != inputs.size())
    {
        THROW_IE_EXCEPTION << "Function inputs number differ from number of given inputs";
    }

    size_t i = 0;
    for (const auto& it : input_info)
    {
        shared_ptr<runtime::ie::IETensor> tv =
            static_pointer_cast<runtime::ie::IETensor>(inputs[i]);
        size_t size = tv->get_element_count();
        const float* orig_data = static_cast<const float*>(tv->get_data_ptr());
        infer_request.SetBlob(it.first,
                              fill_blob(it.second->getTensorDesc().getDims(), orig_data, size));
        i++;
    }

    //  Prepare output blobs
    string output_name = m_network.getOutputsInfo().begin()->first;

    infer_request.Infer();
    InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);

    InferenceEngine::MemoryBlob::Ptr moutput =
        InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
    if (!moutput)
    {
        THROW_IE_EXCEPTION << "Cannot get output MemoryBlob in call_with_validate()";
    }

    auto lm = moutput->rmap();
    float* output_ptr = lm.as<float*>();
    size_t size = shape_size(output->getTensorDesc().getDims());
    outputs[0]->write(output_ptr, size * sizeof(float));
    return true;
}

InferenceEngine::Blob::Ptr
    runtime::ie::fill_blob(InferenceEngine::SizeVector shape, const float* data, size_t data_size)
{
    InferenceEngine::Layout layout;
    switch (shape.size())
    {
    case 1: layout = InferenceEngine::Layout::C; break;
    case 2: layout = InferenceEngine::Layout::NC; break;
    case 3: layout = InferenceEngine::Layout::CHW; break;
    case 4: layout = InferenceEngine::Layout::NCHW; break;
    case 5: layout = InferenceEngine::Layout::NCDHW; break;
    default: THROW_IE_EXCEPTION << "Can't convert dims " << shape.size() << " to Layout!";
    }
    InferenceEngine::MemoryBlob::Ptr blob(
        new InferenceEngine::TBlob<float>({InferenceEngine::Precision::FP32, shape, layout}));
    blob->allocate();
    float* blob_ptr = blob->rwmap().as<float*>();
    memcpy(blob_ptr, data, data_size * sizeof(float));
    return blob;
}
