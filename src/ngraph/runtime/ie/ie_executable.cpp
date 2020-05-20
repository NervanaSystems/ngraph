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

#include <unistd.h>
#include "ngraph/op/constant.hpp"

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

static void print(std::vector <int> const &a) {
   std::cout << "The vector elements are : ";
   
   for(int i=0; i < a.size(); i++)
      std::cout << a.at(i) << ' ';
}

runtime::ie::IE_Executable::IE_Executable(shared_ptr<Function> func, string device)
    : m_device{device}
{

    // Bani
    set_parameters_and_results(*func);

    cout << "\nAfter set_parameters_and_results with initial ngfunc, m_results & m_parameters = " << ", " << func->get_friendly_name() << endl;
    for (auto aNodeShPtr : m_results) { std::cout << aNodeShPtr->get_name() << " (" << aNodeShPtr->get_type_name() << "), "; } std::cout << "\n";
    for (auto aNodeShPtr : m_parameters) { std::cout << aNodeShPtr->get_name() << " (" << aNodeShPtr->get_type_name() << "), "; } std::cout << "\n";



#if 0
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
#endif

#ifdef NGRAPH_DEBUG_ENABLE
    cout << "Nodes in test: ";
    for (const auto& node : func->get_ops())
    {
        cout << node << "  ||  ";
    }
    cout << endl;
    sleep(0.1);
#endif

    // Bani
    std::cout << "\nIn (ngcore repo) IE_Executable constructor BEFORE CNN (after Opset1Upgrade), func=" << func->get_friendly_name() << ", output_size=" << func->get_output_size() << " ==>>\n";
    for (auto aNodeShPtr : func->get_ordered_ops()) { std::cout << aNodeShPtr->get_name() << " (" << aNodeShPtr->get_type_name() << "), "; } std::cout << "\n";
    sleep(0.1);

    m_network = InferenceEngine::CNNNetwork(func);

    // Bani - serialize the CNNNetwork
    #if 0
    m_network.serialize("bani_cnn5.xml", "bani_cnn5.bin");
    std::cout << "\nIE_Executable m_network serialized. bani_cnn5.xml bani_cnn5.bin, " << m_network.getFunction()->get_friendly_name() << "\n"; // Bani
    #endif
    #if 1
    // Or try, std::string fileprefix = std::string("bani_cnn5") + "_" + m_network.getFunction()->get_friendly_name();
    std::string fileprefix = std::string("bani_cnn5") + "_" + m_network.getFunction()->get_friendly_name();
    /*
    std::ostringstream oss_fileprefix;
    oss_fileprefix << "bani_cnn5" + "_" + m_network.getFunction()->get_friendly_name();
    std::string fileprefix = oss_fileprefix.str()
    */
    //m_network.serialize("bani_cnn5.xml", "bani_cnn5.bin");
    m_network.serialize(fileprefix + ".xml", fileprefix + ".bin");
    //std::cout << "    IE_Executable m_network serialized. bani_cnn5.xml bani_cnn5.bin, " << m_network.getFunction()->get_friendly_name() << "\n"; // Bani
    std::cout << "\nIE_Executable m_network serialized. " << fileprefix << "\n"; // Bani
    #endif
    
    //set_parameters_and_results(*func);
    //set_parameters_and_results(*(m_network.getFunction()));


    // Bani
    std::cout << "\nIn (ngcore repo) IE_Executable constructor AFTER CNN, m_network.getFunction()=" << m_network.getFunction()->get_friendly_name() << ", output_size=" << m_network.getFunction()->get_output_size() << " ==>>\n";
    for (auto aNodeShPtr : m_network.getFunction()->get_ordered_ops()) { std::cout << aNodeShPtr->get_name() << " (" << aNodeShPtr->get_type_name() << "), "; } std::cout << "\n\n";
    sleep(0.1);


    // Eliminate Constant nodes that are directly connected to an Output/Result node
    for (const auto& node : m_network.getFunction()->get_ops()) {
        //std::cout << "    trying fn(name) = " << node->get_friendly_name() << "(" << node->get_name() << ", " << node->get_type_name() << ")\n";
        if (node->is_constant()) {
            //std::cout << "    node is_constant, fn(name) = " << node->get_friendly_name() << "(" << node->get_name() << ", " << node->get_type_name() << ")\n";
            auto outputs = node->outputs(); // std::vector<Output<Node>>
            if(outputs.size() == 1) {
                auto const& outHndl = outputs[0];
                auto const& targetInputHndls = outHndl.get_target_inputs(); // std::set<Input<Node>> get_target_inputs()
                //std::cout << "        node's outHndl.get_target_inputs().size() = " << targetInputHndls.size() << "  ==>> "; for (auto x : targetInputHndls) { std::cout << x.get_node()->get_friendly_name() << " (" << x.get_node()->get_name() << " / " << x.get_node()->get_type_name() << "), "; } std::cout << "\n";
                // TODO: put a check: if we get size > 1, it's Ok as long as all the target inputs have same friendly_name (but diff actual name)
                if(targetInputHndls.size() > 0) {
                    auto const& outNode = targetInputHndls.begin()->get_node();
                    //std::cout << "        node's outputs.size() == ?, fn(name) = " << outNode->get_friendly_name() << "(" << outNode->get_name() << ", " << outNode->get_type_name() << ")\n";
                    if(outNode->is_output()) {
                        // (input-const, output-result) e.g. Constant_675->Result_352, ngraph_output_1->Result_350
                        m_nongraph_const_outputs.insert(std::pair<std::string, std::string>(node->get_friendly_name(), outNode->get_friendly_name()));
                    }
                }
            }
        }
    }
    cout << "\nAfter m_nongraph_const_outputs ==> "; for (auto const& pair : m_nongraph_const_outputs) { std::cout << pair.first << "->" << pair.second << ", "; } std::cout << "\n";
    // Example: Constant_673->Result_353, Constant_675->Result_352, ngraph_output_1->Result_350, ngraph_output_2->Result_351, 
    // ngraph_output_6->Result_355, ngraph_output_7->Result_356, ngraph_output_8->Result_357, ngraph_output_9->Result_358,


#if 1
    // We need to save the mapping from CNN network's Result nodes to the original NG-function's result-contributing nodes
    for(auto aNodeShPtr : m_results) {
        m_map_result_to_ngnode.insert(std::pair<std::string, std::string>(aNodeShPtr->get_friendly_name(), "UNKNOWN"));
    }
    for (const auto& node : m_network.getFunction()->get_ops()) {
        std::cout << "    trying fn(name) = " << node->get_friendly_name() << "(" << node->get_name() << ", " << node->get_type_name() << ")\n";
        auto outputs = node->outputs(); // std::vector<Output<Node>>
        if(outputs.size() == 1) {
            auto const& outHndl = outputs[0];
            auto const& targetInputHndls = outHndl.get_target_inputs(); // std::set<Input<Node>> get_target_inputs()
            //std::cout << "        node's outHndl.get_target_inputs().size() = " << targetInputHndls.size() << "  ==>> "; for (auto x : targetInputHndls) { std::cout << x.get_node()->get_friendly_name() << " (" << x.get_node()->get_name() << " / " << x.get_node()->get_type_name() << "), "; } std::cout << "\n";
            // TODO: put a check: if we get size > 1, it's Ok as long as all the target inputs have same friendly_name (but diff actual name)
            if(targetInputHndls.size() > 0) {
                auto const& outNode = targetInputHndls.begin()->get_node();
                //std::cout << "        node's outputs.size() == ?, fn(name) = " << outNode->get_friendly_name() << "(" << outNode->get_name() << ", " << outNode->get_type_name() << ")\n";
                if(m_map_result_to_ngnode.count(outNode->get_friendly_name()) == 0) {
                    continue; // we are not interested, as it is not a Result_ node
                }
                if(outNode->is_output()) {
                    // (result, from) e.g. Result_353->Constant_673, Result_350->ngraph_output_1
                    m_map_result_to_ngnode.erase(outNode->get_friendly_name());
                    m_map_result_to_ngnode.insert(std::pair<std::string, std::string>(outNode->get_friendly_name(), node->get_friendly_name()));
                }
            }
        }
    }
    cout << "\nAfter m_map_result_to_ngnode ==> "; for (auto const& pair : m_map_result_to_ngnode) { std::cout << pair.first << "->" << pair.second << ", "; } std::cout << "\n";
#endif


    InferenceEngine::Core ie;
    //  Load model to the plugin (BACKEND_NAME)
    InferenceEngine::ExecutableNetwork exe_network = ie.LoadNetwork(m_network, m_device);

    std::cout << "\nIn (ngcore repo) IE_Executable constructor AFTER LoadNetwork, InferenceEngine::ExecutableNetwork" << ", " << func->get_friendly_name() << " ==>\n";
    //for (auto aNodeShPtr : m_network.getFunction()->get_ordered_ops()) { std::cout << aNodeShPtr->get_name() << " (" << aNodeShPtr->get_type_name() << "), "; } std::cout << "\n\n";
    std::cout << "    outs = "; for (auto const& pair : exe_network.GetOutputsInfo()) { std::cout << pair.first << ", "; } std::cout << "\n";
    std::cout << "    ins = "; for (auto const& pair : exe_network.GetInputsInfo()) { std::cout << pair.first << ", "; } std::cout << "\n";
    sleep(0.1);

    //  Create infer request
    std::cout << "\nIn (ngcore repo) IE_Executable constructor BEFORE exe_network.CreateInferRequest()... " << ", " << func->get_friendly_name() << "\n";
    m_infer_req = exe_network.CreateInferRequest();
    std::cout << "\nIn (ngcore repo) IE_Executable constructor AFTER exe_network.CreateInferRequest()... " << ", " << func->get_friendly_name() << "\n\n";

}

bool runtime::ie::IE_Executable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                      const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // Example from ng func (the param vectors are in this order for outputs and inputs respectively):
    //     outs(10) = Add_359(*), Constant_673, Constant_675, ngraph_output_0(*), ngraph_output_1, ngraph_output_2, ngraph_output_6, ngraph_output_7, ngraph_output_8, ngraph_output_9,
    //         The ones with * are actual dynamic computed results/outs from the IE/CNN
    //     ins(2) = _arg_import/best_practices_0_0, _arg_import/read_tensor_0_1,
    // Example Const to ng result map (m_nongraph_const_outputs), helpful for tensor data copying:
    //     Constant_673->Result_353, Constant_675->Result_352, ngraph_output_1->Result_350, ngraph_output_2->Result_351, 
    //     ngraph_output_6->Result_355, ngraph_output_7->Result_356, ngraph_output_8->Result_357, ngraph_output_9->Result_358,
    std::cout << "\nIn (ngcore repo) BEGIN runtime::ie::IE_Executable::call ... " << ", " << m_network.getFunction()->get_friendly_name() << "\n";

    #if 0
    InferenceEngine::Core ie;

    //  Loading model to the plugin (BACKEND_NAME)
    InferenceEngine::ExecutableNetwork exe_network = ie.LoadNetwork(m_network, m_device);
    std::cout << "In (ngcore repo) IE_Executable::call(...) AFTER ie.LoadNetwork, func/m_network.getFunction()=" << m_network.getFunction()->get_friendly_name() << ", output_size=" << m_network.getFunction()->get_output_size() << " ==>>\n";
    for (auto aNodeShPtr : m_network.getFunction()->get_ordered_ops()) { std::cout << aNodeShPtr->get_name() << ","; } std::cout << "\n";

    //  Create infer request
    InferenceEngine::InferRequest infer_request = exe_network.CreateInferRequest();
    m_infer_req = infer_request;
    #endif 

    //  Prepare input and output blobs
    InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();

    if (input_info.size() != inputs.size())
    {
        THROW_IE_EXCEPTION << "Function inputs number differ from number of given inputs";
    }

    size_t i = 0;
    for (const auto& it : input_info)
    {
        std::cout << "    runtime::ie::IE_Executable::call, input iterator = " << it.first << ", " << m_network.getFunction()->get_friendly_name() << "\n";
        shared_ptr<runtime::ie::IETensor> tv =
            static_pointer_cast<runtime::ie::IETensor>(inputs[i]);
        m_infer_req.SetBlob(it.first,
                              fill_blob(it.second->getTensorDesc().getDims(),
                                        tv->get_data_ptr(),
                                        tv->get_element_count(),
                                        tv->get_element_type()));
        i++;
    }


    // Bani DEBUG
    //shared_ptr<Function> func = m_network.getFunction();
    //std::cout << "\n\nIn (ngcore repo) IE_Executable::call BEFORE Infer(), func=" << func->get_friendly_name() << ", output_size=" << func->get_output_size() << " ==>>\n";
    //for (auto aNodeShPtr : func->get_ordered_ops()) { std::cout << aNodeShPtr->get_name() << ","; } std::cout << "\n";
    //sleep(0.1);
    std::cout << "\n\nIn (ngcore repo) IE_Executable::call BEFORE Infer()" << ", " << m_network.getFunction()->get_friendly_name() << "\n";

    m_infer_req.Infer();

    std::cout << "\nIn (ngcore repo) IE_Executable::call AFTER Infer()" << ", " << m_network.getFunction()->get_friendly_name() << "\n";
    sleep(0.1);

    //  Prepare output blobs
    //string output_name = m_network.getOutputsInfo().begin()->first;
    
    //DEBUG Bani - o is shared_ptr<runtime::Tensor>
    std::cout << "IE_Executable::call outputs tensors"<< ", " << m_network.getFunction()->get_friendly_name() << " ==> "; for(auto o : outputs) { std::cout << o->get_name() << " / " << o->get_size_in_bytes() << ",  "; } std::cout << "\n";

    // Example: same numbers of items in each (e.g. 10)
    // m_network.getOutputsInfo() => Add_359(*), Constant_673, Constant_675, ngraph_output_0(*), ngraph_output_1, ngraph_output_2, ngraph_output_6, ngraph_output_7, ngraph_output_8, ngraph_output_9
    // m_results = Result_349 (0), Result_350 (1), Result_351 (2), Result_352 (3), Result_353 (4), Result_354 (5), Result_355 (6), Result_356 (7), Result_357 (8), Result_358 (9),
    // Also, order of Output tensors follow the order of m_results, but *NOT* the order of m_network.getOutputsInfo()
    NGRAPH_CHECK(m_results.size()==m_network.getOutputsInfo().size(), "Mismatching number of output items");
    NGRAPH_CHECK(m_results.size()==outputs.size(), "Mismatching number of output items between tensors (", outputs.size(), ") and results (", m_results.size() ,")");
    
    int idx = 0;
    //for(auto const& pair : m_network.getOutputsInfo()) {
    for(auto aNodeShPtr : m_results) {
        string ng_result_name = aNodeShPtr->get_name(); // e.g. Result_350
        if(m_map_result_to_ngnode.find(ng_result_name) == m_map_result_to_ngnode.end()) {
                THROW_IE_EXCEPTION << "\n!!! Cannot locate in m_map_result_to_ngnode, ng_result_name = " << ng_result_name << " !!!\n";
        }
        string output_name = m_map_result_to_ngnode.at(ng_result_name); // e.g. Constant_673

        std::cout << "\nIn (ngcore repo) IE_Executable::call Prepare output blobs, output_name = " << output_name << ", ng_result_name = " << ng_result_name <<
            ", idx=" << idx << ", tensor outputs[idx].size()=" << outputs[idx]->get_size_in_bytes() << "\n"; // Bani

        // Now, e.g. in first iteration, output_name Add_359 is matched with ng_result_name Result_349, both at idx = 0

        if(m_nongraph_const_outputs.find(output_name) != m_nongraph_const_outputs.end()) {
            // So, we will shortcut the Const value into the Result_ node
            std::cout << "    found Constant node in m_nongraph_const_outputs: " << output_name << "\n";
            bool found = false;
            for(const auto& node : m_network.getFunction()->get_ops()) {
                if(node->get_friendly_name().compare(output_name) == 0) { // Example output_name: Constant_673
                    if(node->is_constant()) {
                        found = true;
                        const ngraph::op::Constant* const_node = dynamic_cast<const ngraph::op::Constant*>(&(*node));
                        if(const_node) {
                            auto num_bytes = shape_size(const_node->get_shape()) * const_node->get_element_type().size();
                            std::cout << "    getting data from Const node, num_bytes=" << num_bytes << "\n";
                            const void* value = const_node->get_data_ptr();
                            outputs[idx]->write(value, num_bytes);
                        } else {
                            THROW_IE_EXCEPTION << "\n!!! Cannot dynamic_cast<const ngraph::op::Constant*>, const-node = " << output_name << " !!!\n";
                        }
                        break;
                    }
                }
            }
            if(!found) {
                THROW_IE_EXCEPTION << "\n!!! Cannot locate in m_network.getFunction(), const-node = " << output_name << " !!!\n";
            }

            idx++;
            continue;
        }

        std::cout << "    getting data from IE/CNN blob" << ", " << m_network.getFunction()->get_friendly_name() << "\n";

        InferenceEngine::Blob::Ptr output = m_infer_req.GetBlob(output_name);
        std::cout << "    m_infer_req.GetBlob(output_name) done.\n";

        InferenceEngine::MemoryBlob::Ptr moutput =
            InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
        if (!moutput)
        {
            THROW_IE_EXCEPTION << "Cannot get output MemoryBlob in call_with_validate()";
        }

        auto lm = moutput->rmap();
        uint8_t* output_ptr = lm.as<uint8_t*>();
        //outputs[0]->write(output_ptr, moutput->byteSize());
        outputs[idx]->write(output_ptr, moutput->byteSize());
        idx++;
    }

    std::cout << "\nIn (ngcore repo) END runtime::ie::IE_Executable::call" << ", " << m_network.getFunction()->get_friendly_name() << "...\n\n\n\n";
    return true;
}
