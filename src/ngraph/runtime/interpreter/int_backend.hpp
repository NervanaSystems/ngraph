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

#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class INTBackend;
            class INTExecutable;
            class INTBackendConstructor;
        }
    }
}

class ngraph::runtime::interpreter::INTBackend : public Backend
{
public:
    INTBackend();
    INTBackend(const std::vector<std::string>& unsupported_op_name_list);
    INTBackend(const INTBackend&) = delete;
    INTBackend(INTBackend&&) = delete;
    INTBackend& operator=(const INTBackend&) = delete;

    std::shared_ptr<Tensor>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<Tensor> create_tensor(const element::Type& type, const Shape& shape) override;

    std::shared_ptr<Executable> compile(std::shared_ptr<Function> function,
                                        bool enable_performance_data = false) override;

    bool is_supported(const Node& node) const override;

private:
    std::set<std::string> m_unsupported_op_name_list;
};

class ngraph::runtime::interpreter::INTBackendConstructor : public BackendConstructor
{
public:
    std::shared_ptr<Backend> create(const std::string& config) override
    {
        return std::make_shared<INTBackend>();
    }
};
