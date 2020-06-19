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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_core.hpp>
#include "ngraph/runtime/executable.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ie
        {
            // A Inference Engine executable object produced by compiling an nGraph function.
            class IE_Executable final : public Executable
            {
            public:
                IE_Executable(std::shared_ptr<Function> func, std::string device);
                virtual ~IE_Executable() {}
                bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                          const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) final;

            private:
                InferenceEngine::CNNNetwork m_network;
                InferenceEngine::InferRequest m_infer_req;
                std::string m_device;
                std::map<std::string, int> m_map_cnnparam_to_tfidx; // which CNN param maps to which index of the TF input tensor
                std::map<std::string, int> m_map_cnnresult_to_tfidx; // which CNN result maps to which index of the TF output tensor
                std::map<std::string, void*> m_map_cnnconstresult_to_ngnodeptr;
                std::map<std::string, std::string> m_nongraph_const_outputs; // (input-const, output-result)
                std::map<std::string, std::string> m_map_result_to_ngnode; // (result, from) e.g. Result_353->Constant_673, Result_350->ngraph_output_1
                std::map<std::string, void*> m_map_result_to_ngnodeptr; // same as above one
            };
        }
    }
}
