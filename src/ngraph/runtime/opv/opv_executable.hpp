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

#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ie_core.hpp>


#include "ngraph/ops.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"

#include "ngraph/runtime/opv/opv_tensor.hpp"



namespace ngraph
{
    namespace runtime
    {
        namespace opv
        {
            class OPVBackend;
            class OPVExecutable;

        } // namespace opv
    }     // namespace runtime
} // namespace ngraph

class ngraph::runtime::opv::OPVExecutable : public Executable
{
    friend class OPVBackend;

public:
    OPVExecutable(const std::shared_ptr<Function>& function,
                  bool enable_performance_collection = false);

    bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& inputs) override;

    virtual void save(std::ostream& output_stream) override;

protected:

    bool m_performance_counters_enabled = false;
    InferenceEngine::CNNNetwork network;
    std::string device; // TODO: for now its CPU? figure it out later

private:
    InferenceEngine::Blob::Ptr fill_blob(InferenceEngine::SizeVector shape, float*, size_t);
}; 