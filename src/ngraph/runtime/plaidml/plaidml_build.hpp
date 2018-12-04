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

#include <plaidml/plaidml++.h>

#include <string>
#include <unordered_map>

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
            class Compiler;
            struct TensorInfo;

            enum class TensorContents
            {
                DATA = 0,
                LOGICAL = 1
            };
        }
    }
}

// Holds information about a particular tensor.
struct ngraph::runtime::plaidml::TensorInfo final
{
    TensorInfo(vertexai::plaidml::variable _var, TensorContents _contents)
        : var{std::move(_var)}
        , contents{_contents}
    {
    }

    vertexai::plaidml::variable var;
    TensorContents contents;
};

// Holds the intermediate state of a function compilation.
struct ngraph::runtime::plaidml::Build final
{
    Config* config = nullptr;
    Compiler* compiler = nullptr;
    std::shared_ptr<Function> func;
    std::unordered_map<descriptor::Tensor*, std::string> input_names;
    std::unordered_map<descriptor::Tensor*, std::string> output_names;
    vertexai::plaidml::compose composer;
    std::unordered_map<descriptor::Tensor*, TensorInfo> bindings;
    bool io_dim_override = false;
    std::size_t io_dim_override_count = 0;
};
