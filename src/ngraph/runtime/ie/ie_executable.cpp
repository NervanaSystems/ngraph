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

namespace
{
    InferenceEngine::Blob::Ptr fill_blob(InferenceEngine::SizeVector shape,
                                         const void* data,
                                         size_t data_size,
                                         const element::Type& elem_type)
    {
        InferenceEngine::Layout layout;
        switch (shape.size())
        {
        case 0: layout = InferenceEngine::Layout::SCALAR; break;
        case 1: layout = InferenceEngine::Layout::C; break;
        case 2: layout = InferenceEngine::Layout::NC; break;
        case 3: layout = InferenceEngine::Layout::CHW; break;
        case 4: layout = InferenceEngine::Layout::NCHW; break;
        case 5: layout = InferenceEngine::Layout::NCDHW; break;
        case 6: layout = InferenceEngine::Layout::GOIDHW; break;
        default: THROW_IE_EXCEPTION << "Can't convert dims " << shape.size() << " to Layout!";
        }

        InferenceEngine::MemoryBlob::Ptr blob;

#define MAKE_IE_TBLOB(type_, precision_, shape_, layout_)                                          \
    make_shared<InferenceEngine::TBlob<type_>>(                                                    \
        InferenceEngine::TensorDesc{InferenceEngine::Precision::precision_, shape_, layout_})

        switch (elem_type)
        {
        case element::Type_t::f32: blob = MAKE_IE_TBLOB(float, FP32, shape, layout); break;
        case element::Type_t::i16: blob = MAKE_IE_TBLOB(int16_t, I16, shape, layout); break;
        case element::Type_t::u8: blob = MAKE_IE_TBLOB(uint8_t, U8, shape, layout); break;
        case element::Type_t::i8: blob = MAKE_IE_TBLOB(int8_t, I8, shape, layout); break;
        case element::Type_t::u16: blob = MAKE_IE_TBLOB(uint16_t, U16, shape, layout); break;
        case element::Type_t::i32: blob = MAKE_IE_TBLOB(int32_t, I32, shape, layout); break;
        case element::Type_t::i64: blob = MAKE_IE_TBLOB(int64_t, I64, shape, layout); break;
        case element::Type_t::u64: blob = MAKE_IE_TBLOB(uint64_t, U64, shape, layout); break;
        case element::Type_t::boolean: blob = MAKE_IE_TBLOB(uint8_t, BOOL, shape, layout); break;
        default: THROW_IE_EXCEPTION << "Can't convert type " << elem_type << " to IE Precision!";
        }
#undef MAKE_IE_TBLOB

        blob->allocate();
        uint8_t* blob_ptr = blob->rwmap().as<uint8_t*>();
        memcpy(blob_ptr, data, data_size * elem_type.size());
        return blob;
    }
}

namespace
{
    std::set<NodeTypeInfo> get_ie_ops()
    {
        std::set<NodeTypeInfo> ie_ops = get_opset1().get_type_info_set();
        auto& opset2 = get_opset2().get_type_info_set();
        ie_ops.insert(opset2.begin(), opset2.end());
        auto& opset3 = get_opset3().get_type_info_set();
        ie_ops.insert(opset3.begin(), opset3.end());
        return ie_ops;
    }
}

runtime::ie::IE_Executable::IE_Executable(shared_ptr<Function> func, string device)
    : m_device{device}
{
    static std::set<NodeTypeInfo> ie_ops = get_ie_ops();
    pass::Manager passes;
    passes.register_pass<pass::Opset1Upgrade>();
    passes.run_passes(func);

    for (const auto& node : func->get_ops())
    {
        if (ie_ops.find(node->get_type_info()) == ie_ops.end())
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
        infer_request.SetBlob(it.first,
                              fill_blob(it.second->getTensorDesc().getDims(),
                                        tv->get_data_ptr(),
                                        tv->get_element_count(),
                                        tv->get_element_type()));
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
    uint8_t* output_ptr = lm.as<uint8_t*>();
    outputs[0]->write(output_ptr, moutput->byteSize());
    return true;
}
