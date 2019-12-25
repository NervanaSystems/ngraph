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
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <plaidml/plaidml++.h>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/plaidml/plaidml_config.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            struct Build;
            class PlaidML_Executable;
        }
    }
}

// A PlaidML executable object produced by compiling an nGraph function.
class ngraph::runtime::plaidml::PlaidML_Executable final : public Executable
{
public:
    PlaidML_Executable(Build build, std::shared_ptr<Function> func);
    virtual ~PlaidML_Executable() {}
    bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) final;

    std::vector<PerformanceCounter> get_performance_data() const final;

    void save_as_format(const std::string& filename, plaidml_file_format format) const;

    const std::shared_ptr<Function>& src_func() const { return m_src_func; }
private:
    mutable std::mutex m_mu; // Locks the invoker while scheduling invocations.
    mutable bool m_bound = false;
    Config* m_config;
    std::shared_ptr<Function> m_func;
    std::shared_ptr<Function> m_src_func; // The original source function.
    std::unordered_map<descriptor::Tensor*, std::string> m_input_names;
    std::unordered_map<descriptor::Tensor*, std::string> m_output_names;
    mutable std::vector<std::weak_ptr<runtime::Tensor>> m_bound_inputs;
    mutable std::vector<std::weak_ptr<runtime::Tensor>> m_bound_outputs;
    mutable vertexai::plaidml::invoker m_invoker;
};
