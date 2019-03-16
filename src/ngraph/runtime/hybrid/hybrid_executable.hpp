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

#include "ngraph/runtime/executable.hpp"

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

class ngraph::runtime::hybrid::HybridExecutable : public runtime::Executable
{
public:
    HybridExecutable(const std::vector<std::shared_ptr<runtime::Backend>>& backend_list,
                     const std::shared_ptr<Function>& func,
                     bool enable_performance_collection = false,
                     bool debug_enabled = false);

    bool call(const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& inputs) override;

private:
    std::shared_ptr<ngraph::Function> m_function;
    std::shared_ptr<Executable> m_executable;
    std::unordered_map<std::shared_ptr<ngraph::op::Parameter>, std::shared_ptr<ngraph::op::Result>>
        m_map_parameter_to_result;

    std::vector<std::shared_ptr<runtime::Backend>> m_backend_list;
    bool m_debug_enabled = false;

    size_t get_placement(const runtime::Tensor* t);
};
