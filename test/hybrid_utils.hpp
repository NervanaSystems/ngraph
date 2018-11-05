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

#include "ngraph/ngraph.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

class TestBackend : public ngraph::runtime::Backend
{
public:
    TestBackend(const std::vector<std::shared_ptr<ngraph::runtime::Backend>>& backend_list);

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type,
                      const ngraph::Shape& shape) override;

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type,
                      const ngraph::Shape& shape,
                      void* memory_pointer) override;

    bool compile(std::shared_ptr<ngraph::Function> func) override;

    bool call(std::shared_ptr<ngraph::Function> func,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& inputs) override;

private:
    // This list of backends is in order of priority with the first backend higher priority
    // than the second.
    std::vector<std::shared_ptr<ngraph::runtime::Backend>> m_backend_list;

protected:
    class FunctionInstance
    {
    public:
        std::shared_ptr<ngraph::Function> m_function;
        std::vector<std::shared_ptr<ngraph::Function>> m_sub_functions;
        std::unordered_map<std::shared_ptr<ngraph::op::Parameter>,
                           std::shared_ptr<ngraph::op::Result>>
            m_map_parameter_to_result;
    };

    std::map<std::shared_ptr<ngraph::Function>, FunctionInstance> m_function_map;
};

class BackendWrapper : public ngraph::runtime::Backend
{
public:
    BackendWrapper(const std::string& backend_name,
                   const std::set<std::string>& supported_ops,
                   const std::string& name);

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type,
                      const ngraph::Shape& shape) override;

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type,
                      const ngraph::Shape& shape,
                      void* memory_pointer) override;

    bool compile(std::shared_ptr<ngraph::Function> func) override;

    bool call(std::shared_ptr<ngraph::Function> func,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& inputs) override;

    bool is_supported(const ngraph::Node& node) const override;

private:
    std::shared_ptr<ngraph::runtime::Backend> m_backend;
    const std::set<std::string> m_supported_ops;
    const std::string m_name;
};
