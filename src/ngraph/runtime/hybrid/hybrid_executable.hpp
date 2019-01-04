//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace hybrid
        {
            class HybridExecutable;
        }
    }
}

class ngraph::runtime::hybrid::HybridExecutable : public ngraph::runtime::Executable
{
    friend class HybridBackend;

public:
    HybridExecutable();

    bool execute(const std::vector<runtime::Tensor*>& outputs,
                 const std::vector<runtime::Tensor*>& inputs) override;

private:
    HybridExecutable(const std::vector<std::shared_ptr<runtime::Backend>>& backend_list,
                     std::shared_ptr<Function> function,
                     bool enable_performance_collection);
    std::shared_ptr<ngraph::Function> m_function;
    std::vector<std::shared_ptr<ngraph::Function>> m_sub_functions;
    std::unordered_map<std::shared_ptr<ngraph::op::Parameter>, std::shared_ptr<ngraph::op::Result>>
        m_map_parameter_to_result;
    std::unordered_map<std::shared_ptr<Function>, runtime::SharedHandle> m_handle_map;
    std::vector<std::shared_ptr<runtime::Backend>> m_backend_list;
};
