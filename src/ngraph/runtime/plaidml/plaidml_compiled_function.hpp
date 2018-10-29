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

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <plaidml/plaidml++.h>

#include "ngraph/function.hpp"
#include "ngraph/runtime/plaidml/plaidml_config.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            struct Build;
            class CompiledFunction;
        }
    }
}

// A PlaidML compiled function object produced by compiling an nGraph function.
class ngraph::runtime::plaidml::CompiledFunction final
{
public:
    CompiledFunction(Build build);

    bool schedule_invocation(const std::vector<std::shared_ptr<runtime::Tensor>>& inputs,
                             const std::vector<std::shared_ptr<runtime::Tensor>>& outputs) const;

    void save(const std::string& filename, plaidml_file_format format) const;

private:
    mutable std::mutex m_mu; // Locks the invoker while scheduling invocations.
    mutable bool m_bound = false;
    Config* m_config;
    std::shared_ptr<Function> m_func;
    std::unordered_map<descriptor::Tensor*, std::string> m_input_names;
    std::unordered_map<descriptor::Tensor*, std::string> m_output_names;
    mutable vertexai::plaidml::invoker m_invoker;
};
