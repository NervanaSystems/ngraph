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

#include "ngraph/runtime/opv/opv_executable.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/ops.hpp"

#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/util.hpp"

#include "ngraph/runtime/opv/opv_executable.hpp"


using namespace std;
using namespace ngraph;
using namespace InferenceEngine;

using descriptor::layout::DenseTensorLayout;


runtime::opv::OPVExecutable::OPVExecutable(const shared_ptr<Function>& function,
                  bool enable_performance_collection)
{
    // OPV backend can handle only opset 1, hence running upgrade pass
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(function);


    // From here: https://github.com/NervanaSystems/ngraph/blob/master/test/util/backend_utils.hpp#L84
    auto opset = ngraph::get_opset1();
    bool all_opset1 = true;
    for (auto& node : function->get_ops())
    {
        if (!opset.contains_op_type(node.get()))
        {
            all_opset1 = false;
            break;
        }
    }

    if (!all_opset1)
    {
        std::cout << "UNSUPPORTED OPS DETECTED!" << std::endl;
        THROW_IE_EXCEPTION << "Exit from test";
    }
    std::cout << "Nodes in test: ";
    for (auto& node : function->get_ops())
    {
        std::cout << node->get_type_info().name << " ";
    }
    std::cout << std::endl;
    network = InferenceEngine::CNNNetwork(function);
    device = "CPU"; // TODO: fix later
}

bool runtime::opv::OPVExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                               const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // from here: https://github.com/NervanaSystems/ngraph/blob/fd32fbbe7e72bfcf56688b22e57591f519456a46/src/ngraph/runtime/interpreter/int_executable.cpp#L107
    // Converting to HostTensor since it exposes get_data_ptr, which we will need to populate orig_data
    vector<shared_ptr<HostTensor>> func_inputs_as_host_tensors;
    for (auto tensor : inputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_inputs_as_host_tensors.push_back(host_tensor);
    }

    // From here: https://github.com/NervanaSystems/ngraph/blob/master/test/util/backend_utils.hpp#L113
    try
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
            THROW_IE_EXCEPTION
                << "Function inputs number differ from number of given inputs";
        }

        size_t i = 0;
        for (auto& it : inputInfo)
        {
            size_t size_in_bytes = inputs[i]->get_size_in_bytes();
            float* orig_data = func_inputs_as_host_tensors[i]->get_data_ptr<float>();
            cout << "HERE. Inputs: " << orig_data[0] << " " << orig_data[1] << "\n";
            // TODO: receiving bad input data here
            inferRequest.SetBlob(it.first,
                                    fill_blob(it.second->getTensorDesc().getDims(), orig_data, (size_in_bytes/sizeof(float))));
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

        // TODO remove
        cout << "HERE:: " << output_ptr[0] << " " << output_ptr[1] << "\n";
        outputs[0]->write(output_ptr, size * sizeof(float));

        return true;
    }
    catch (...)
    {
        THROW_IE_EXCEPTION << "FAILED";
    }

}

void runtime::opv::OPVExecutable::save(ostream& out)
{
    throw std::runtime_error("Save function is unimplemented for opv executable");
}

InferenceEngine::Blob::Ptr runtime::opv::OPVExecutable::fill_blob(InferenceEngine::SizeVector shape, float* data, size_t num_floats)
{
    // From here: https://github.com/NervanaSystems/ngraph/blob/master/test/util/backend_utils.cpp#L26
    Layout layout;
    switch (shape.size())
    {
    case 1: layout = Layout::C; break;
    case 2: layout = Layout::NC; break;
    case 3: layout = Layout::CHW; break;
    case 4: layout = Layout::NCHW; break;
    case 5: layout = Layout::NCDHW; break;
    default: THROW_IE_EXCEPTION << "Can't convert dims " << shape.size() << " to Layout!";
    }
    MemoryBlob::Ptr blob(new TBlob<float>({Precision::FP32, shape, layout}));
    blob->allocate();
    float* blob_ptr = blob->rwmap().as<float*>();
    for (int i = 0; i < num_floats; i++)
    {
        blob_ptr[i] = data[i];
    }
    return blob;
}


// TODO: does executables create tensors? 