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

#include "ngraph/env_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/reshape_sinking.hpp"
#include "ngraph/pass/zero_dim_tensor_elimination.hpp"
#include "ngraph/runtime/ie/ie_executable.hpp"
#include "ngraph/runtime/ie/ie_tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

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

        auto size = data_size * elem_type.size();

#define MAKE_IE_TBLOB(type_, precision_, shape_, layout_)                                          \
    make_shared<InferenceEngine::TBlob<type_>>(                                                    \
        InferenceEngine::TensorDesc{InferenceEngine::Precision::precision_, shape_, layout_},      \
        (type_*)data,                                                                              \
        size)

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
        return blob;
    }
}

runtime::ie::IE_Executable::IE_Executable(shared_ptr<Function> func, string device)
    : m_device{device}
{
    const auto& opset = get_opset1();
    pass::Manager passes;
    passes.register_pass<pass::LikeReplacement>();
    passes.register_pass<pass::NopElimination>();
    passes.register_pass<pass::ZeroDimTensorElimination>();
    passes.register_pass<pass::AlgebraicSimplification>();
    passes.register_pass<pass::ReshapeSinking>();
    passes.register_pass<pass::ReshapeElimination>();
    passes.register_pass<pass::RecurrentReshapeElimination>();
    passes.register_pass<pass::GetOutputElementElimination>();
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

    if (getenv_bool("NGRAPH_IE_DUMP_GRAPHS"))
    {
        auto& name = m_network.getName();
        m_network.serialize(name + ".xml", name + ".bin");
        serialize(name + ".json", func);
    }

    // Bani
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

    // Bani
    // Save the input index mappings from CNN's param name to TF/NGraph's input index
    for (const auto& node : m_network.getFunction()->get_ops()) { // node is a shared_ptr of Node
        if(node->is_parameter()) {
            //const ngraph::op::Parameter* param_nodeptr = dynamic_cast<const ngraph::op::Parameter*>(&(*node));
            //const std::shared_ptr<ngraph::op::Parameter> *param_nodeptr = dynamic_cast<const std::shared_ptr<ngraph::op::Parameter> *>(&node));
            //std::shared_ptr<ngraph::op::Parameter>& param_node = 
            const auto& param_node = as_type_ptr<ngraph::op::Parameter>(node);
            if(param_node) {
                int idx = (int) m_network.getFunction()->get_parameter_index(param_node);
                m_map_cnnparam_to_tfidx.insert(std::pair<std::string, int>(node->get_friendly_name(), idx));
            } else {
                THROW_IE_EXCEPTION << "\n!!! Cannot dynamic_cast parameter node = " << node->get_friendly_name() << " !!!\n";
            }
        }
    }
    cout << "\nIE_Executable ctor, m_map_cnnparam_to_tfidx " << m_network.getFunction()->get_friendly_name() << " ==> "; for (auto const& pair : m_map_cnnparam_to_tfidx) { std::cout << pair.first << "->" << pair.second << ", "; } std::cout << "\n";


    // Bani
    // Save the output index mappings from CNN's result name to TF tensor's output index
    // Example: same numbers of items in each (e.g. 10)
    // m_network.getOutputsInfo() => Add_359(*), Constant_673, Constant_675, ngraph_output_0(*), ngraph_output_1, ngraph_output_2, ngraph_output_6, ngraph_output_7, ngraph_output_8, ngraph_output_9
    // m_results = Result_349 (0), Result_350 (1), Result_351 (2), Result_352 (3), Result_353 (4), Result_354 (5), Result_355 (6), Result_356 (7), Result_357 (8), Result_358 (9),
    // Also, order of Output TF tensors follow the order of m_results, but *NOT* the order of m_network.getOutputsInfo()
    NGRAPH_CHECK(m_results.size()==m_network.getOutputsInfo().size(), "Mismatching number of output items");
    //NGRAPH_CHECK(m_results.size()==outputs.size(), "Mismatching number of output items between tensors (", outputs.size(), ") and results (", m_results.size() ,")");
    int idx = 0;
    for(auto aNodeShPtr : m_results) {
        string ng_result_name = aNodeShPtr->get_name(); // e.g. Result_350
        if(m_map_result_to_ngnode.find(ng_result_name) == m_map_result_to_ngnode.end()) {
                THROW_IE_EXCEPTION << "\n!!! Cannot locate in m_map_result_to_ngnode, ng_result_name = " << ng_result_name << " !!!\n";
        }
        string output_name = m_map_result_to_ngnode.at(ng_result_name); // e.g. Constant_673
        //std::cout << "\nIn IE_Executable::call Prepare output blobs, output_name = " << output_name << ", ng_result_name = " << ng_result_name <<
        //    ", idx=" << idx << ", tensor outputs[idx].size()=" << outputs[idx]->get_size_in_bytes() << "\n"; // Bani

        m_map_cnnresult_to_tfidx.insert(std::pair<std::string, int>(output_name, idx));

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
                            m_map_cnnconstresult_to_node.insert(std::pair<std::string, void*>(output_name, (void*)const_node));
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
        }

        idx++;
    }



    InferenceEngine::Core ie;
    //  Load model to the plugin (BACKEND_NAME)
    InferenceEngine::ExecutableNetwork exe_network = ie.LoadNetwork(m_network, m_device);
    //  Create infer request
    m_infer_req = exe_network.CreateInferRequest();
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
    std::cout << "\nIn BEGIN runtime::ie::IE_Executable::call ... " << ", " << m_network.getFunction()->get_friendly_name() << "\n";

    stopwatch timer;
    //  Prepare input blobs
    timer.start();
    InferenceEngine::InputsDataMap input_info = m_network.getInputsInfo();
    if (input_info.size() != inputs.size())
    {
        THROW_IE_EXCEPTION << "Function inputs number differ from number of given inputs";
    }

    size_t i = 0;
    for (const auto& it : input_info)
    {
        std::cout << "    runtime::ie::IE_Executable::call, input iterator = " << it.first << ", " << m_network.getFunction()->get_friendly_name() << "\n";
        // Check which TF-tensor-input# this it.first matches with
        if(m_map_cnnparam_to_tfidx.find(it.first) == m_map_cnnparam_to_tfidx.end()) {
            THROW_IE_EXCEPTION << "\n!!! Cannot locate in m_map_cnnparam_to_tfidx, input/param = " << it.first << " !!!\n";
        }
        int idx_tensor_input = m_map_cnnparam_to_tfidx.at(it.first);
        if(idx_tensor_input >= inputs.size()) {
            THROW_IE_EXCEPTION << "\n!!! Bad idx_tensor_input for " << it.first << ", idx_tensor_input = " << idx_tensor_input << " !!!\n";
        }

        shared_ptr<runtime::ie::IETensor> tv =
            //static_pointer_cast<runtime::ie::IETensor>(inputs[i]);
            static_pointer_cast<runtime::ie::IETensor>(inputs[idx_tensor_input]); // Bani
        m_infer_req.SetBlob(it.first,
                            fill_blob(it.second->getTensorDesc().getDims(),
                                      tv->get_data_ptr(),
                                      tv->get_element_count(),
                                      tv->get_element_type()));
        i++;
    }
    timer.stop();
    auto time_prep_inputs = timer.get_milliseconds();


    //  Prepare output blobs
    timer.start();
    InferenceEngine::OutputsDataMap output_info = m_network.getOutputsInfo();
    if (output_info.size() != outputs.size())
    {
        THROW_IE_EXCEPTION << "Function outputs number differ from number of given outputs";
    }

    NGRAPH_CHECK(m_results.size()==outputs.size(), "Mismatching number of output items between tensors (", outputs.size(), ") and results (", m_results.size() ,")");
    i = 0;
    for (const auto& it : output_info)
    {
        std::cout << "    runtime::ie::IE_Executable::call, output iterator = " << it.first << ", " << m_network.getFunction()->get_friendly_name() << "\n";
        // Check which TF-tensor-output# this it.first matches with
        if(m_map_cnnresult_to_tfidx.find(it.first) == m_map_cnnresult_to_tfidx.end()) {
            THROW_IE_EXCEPTION << "\n!!! Cannot locate in m_map_cnnresult_to_tfidx, output = " << it.first << " !!!\n";
        }
        int idx_tensor_output = m_map_cnnresult_to_tfidx.at(it.first);
        if(idx_tensor_output >= outputs.size()) {
            THROW_IE_EXCEPTION << "\n!!! Bad idx_tensor_output for " << it.first << ", idx_tensor_output = " << idx_tensor_output << " !!!\n";
        }

        // Check if this is a Constant -> Result sceanrio, in which case we will just short-circuit the value
        auto it2 = m_map_cnnconstresult_to_node.find(it.first);
        if(it2 != m_map_cnnconstresult_to_node.end()) {
            const ngraph::op::Constant* const_node = (ngraph::op::Constant*)(it2->second);
            if(const_node) {
                auto num_bytes = shape_size(const_node->get_shape()) * const_node->get_element_type().size();
                std::cout << "    shortcut data from Const node to TF Output, num_bytes=" << num_bytes << "\n";
                const void* value = const_node->get_data_ptr();
                outputs[idx_tensor_output]->write(value, num_bytes);
                continue;
            } else {
                THROW_IE_EXCEPTION << "\n!!! Cannot get const_node = " << it.first << " !!!\n";
            }
        }

        shared_ptr<runtime::ie::IETensor> tv =
            static_pointer_cast<runtime::ie::IETensor>(outputs[idx_tensor_output]);
        m_infer_req.SetBlob(it.first,
                            fill_blob(it.second->getTensorDesc().getDims(),
                                      tv->get_data_ptr(),
                                      tv->get_element_count(),
                                      tv->get_element_type()));
        i++;
    }
    timer.stop();
    auto time_prep_outputs = timer.get_milliseconds();

    timer.start();
    m_infer_req.Infer();
    timer.stop();
    auto time_infer = timer.get_milliseconds();

    std::cout << "EXEC_CALL_TIMING PROFILE: "
                 << "Prepare Inputs: " << time_prep_inputs << "ms, Prepare Outputs "
                 << time_prep_outputs << "ms, Infer " << time_infer << "ms " << endl;
    
    std::cout << "\nIn END runtime::ie::IE_Executable::call" << ", " << m_network.getFunction()->get_friendly_name() << "...\n\n\n\n";

    return true;
}
