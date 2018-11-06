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

#include "ngraph/runtime/plaidml/plaidml_config.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            class PlaidML_Tensor;
        }
    }
}

class ngraph::runtime::plaidml::PlaidML_Tensor final : public ngraph::runtime::Tensor
{
public:
    PlaidML_Tensor(Config* config,
                   const ngraph::element::Type& element_type,
                   const ngraph::Shape& shape,
                   const std::string& name,
                   void* memory);
    ~PlaidML_Tensor() final {}
    const vertexai::plaidml::tensor<char>& tensor() const { return m_tensor; }
    void write(const void* p, size_t tensor_offset, size_t n) final;
    void read(void* p, size_t tensor_offset, size_t n) const final;

    // Copy the backing memory to the tensor, if needed.
    void sync_input();

    // Copy the tensor to the backing memory, if needed.
    void sync_output();

private:
    vertexai::plaidml::tensor<char> m_tensor;
    void* m_memory;
    size_t m_memory_size;
    bool m_is_logically_zero;
};
