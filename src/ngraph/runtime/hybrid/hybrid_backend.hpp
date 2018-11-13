//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
            class HybridBackend;
        }
    }
}

class ngraph::runtime::hybrid::HybridBackend : public ngraph::runtime::Backend
{
public:
    HybridBackend(
        const std::vector<std::pair<std::string, std::shared_ptr<runtime::Backend>>>& backend_list);

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type,
                      const ngraph::Shape& shape) override;

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type,
                      const ngraph::Shape& shape,
                      void* memory_pointer) override;

    runtime::Handle compile(const std::shared_ptr<ngraph::Function>& func) override;

    bool call(runtime::Handle,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& inputs) override;

    bool is_supported(const ngraph::Node& node) const override;

    const ngraph::op::ParameterVector& get_parameter_descriptors(Handle handle) const override;

    const ngraph::ResultVector& get_result_descriptors(Handle handle) const override;

private:
    class FunctionInstance
    {
    public:
        std::shared_ptr<ngraph::Function> m_function;
        std::vector<std::shared_ptr<ngraph::Function>> m_sub_functions;
        std::unordered_map<std::shared_ptr<ngraph::op::Parameter>,
                           std::shared_ptr<ngraph::op::Result>>
            m_map_parameter_to_result;
    };

    std::vector<std::shared_ptr<FunctionInstance>> m_instances;
    std::vector<std::pair<std::string, std::shared_ptr<runtime::Backend>>> m_backend_list;
};
