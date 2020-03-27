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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "int_backend_visibility.hpp"

#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/runtime/backend.hpp"

namespace interpreter
{
    class INTBackend;
    class INTExecutable;
}

class INTERPRETER_BACKEND_API interpreter::INTBackend : public ngraph::runtime::Backend
{
public:
    INTBackend();
    INTBackend(const std::vector<std::string>& unsupported_op_name_list);
    INTBackend(const INTBackend&) = delete;
    INTBackend(INTBackend&&) = delete;
    INTBackend& operator=(const INTBackend&) = delete;

    std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& type, const ngraph::Shape& shape, void* memory_pointer) override;

    std::shared_ptr<ngraph::runtime::Tensor> create_tensor(const ngraph::element::Type& type, const ngraph::Shape& shape) override;

    std::shared_ptr<ngraph::runtime::Executable> compile(std::shared_ptr<ngraph::Function> function,
                                        bool enable_performance_data = false) override;
    std::shared_ptr<ngraph::runtime::Executable> load(std::istream& input_stream) override;

    bool is_supported(const ngraph::Node& node) const override;

    bool set_config(const std::map<std::string, std::string>& config, std::string& error) override;

private:
    std::set<std::string> m_unsupported_op_name_list;
};
