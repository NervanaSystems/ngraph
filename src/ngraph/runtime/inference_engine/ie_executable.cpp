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

#include "ngraph/runtime/inference_engine/ie_executable.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "ngraph/runtime/inference_engine/ie_tensor_view.hpp"

ngraph::runtime::inference_engine::IE_Executable::IE_Executable(std::shared_ptr<Function> func,
                                                                std::string _device)
{
    auto opset = ngraph::get_opset1();
    pass::Manager passes;
    passes.register_pass<pass::Opset1Upgrade>();
    passes.run_passes(func);

    for (auto& node : func->get_ops())
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
                std::cout << "UNSUPPORTED OP DETECTED: " << node->get_type_info().name << std::endl;
                THROW_IE_EXCEPTION << "Detected op not belonging to opset1!";
            }
        }
    }

#ifdef NGRAPH_DEBUG_ENABLE
    std::cout << "Nodes in test: ";
    for (auto& node : func->get_ops())
    {
        std::cout << node << std::endl;
    }
    std::cout << std::endl;
#endif

    network = InferenceEngine::CNNNetwork(func);
    device = _device == "INFERENCE_ENGINE" ? "CPU" : _device;
    set_parameters_and_results(*func);
}

bool ngraph::runtime::inference_engine::IE_Executable::call(
    const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
    const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    InferenceEngine::Core ie;

    //  Loading model to the plugin (BACKEND_NAME)
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(network, device);
    //  Create infer request
    InferenceEngine::InferRequest inferRequest = exeNetwork.CreateInferRequest();
    //  Prepare input and output blobs
    InferenceEngine::InputsDataMap inputInfo = network.getInputsInfo();

    if (inputInfo.size() != inputs.size())
    {
        THROW_IE_EXCEPTION << "Function inputs number differ from number of given inputs";
    }

    size_t i = 0;
    for (auto& it : inputInfo)
    {
        std::shared_ptr<runtime::inference_engine::IETensorView> tv =
            std::static_pointer_cast<runtime::inference_engine::IETensorView>(inputs[i]);
        size_t size = tv->data.size() / sizeof(float);
        float* orig_data = (float*)tv->data.data();
        std::vector<float> data(orig_data, orig_data + size);
        inferRequest.SetBlob(it.first, fill_blob(it.second->getTensorDesc().getDims(), data));
        i++;
    }

    //  Prepare output blobs
    std::string output_name = network.getOutputsInfo().begin()->first;

    inferRequest.Infer();
    InferenceEngine::Blob::Ptr output = inferRequest.GetBlob(output_name);

    InferenceEngine::MemoryBlob::Ptr moutput =
        InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
    if (!moutput)
    {
        THROW_IE_EXCEPTION << "Cannot get output MemoryBlob in call_with_validate()";
    }

    auto lm = moutput->rmap();
    float* output_ptr = lm.as<float*>();
    // TODO: how to get size without explicit calculation?
    size_t size = 1;
    for (const auto& dim : output->getTensorDesc().getDims())
    {
        size *= dim;
    }
    //  Vector initialization from pointer
    std::vector<float> result(output_ptr, output_ptr + size);
    outputs[0]->write(result.data(), result.size() * sizeof(float));
    return true;
}

InferenceEngine::Blob::Ptr
    ngraph::runtime::inference_engine::fill_blob(InferenceEngine::SizeVector shape,
                                                 std::vector<float> data)
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
    for (int i = 0; i < data.size(); i++)
    {
        blob_ptr[i] = data[i];
    }
    return blob;
}
