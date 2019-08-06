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

#include <memory>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/tensor_value.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class INTTensor;
        }
    }
}

class ngraph::runtime::interpreter::INTTensor : public ngraph::runtime::Tensor
{
public:
    INTTensor(const ngraph::element::Type& element_type,
              const Shape& shape,
              const std::string& name);
    INTTensor(const ngraph::element::Type& element_type, const Shape& shape);
    virtual ~INTTensor() override;

    /// \brief Write bytes directly into the tensor
    /// \param p Pointer to source of data
    /// \param n Number of bytes to write, must be integral number of elements.
    void write(const void* p, size_t n) override;

    /// \brief Read bytes directly from the tensor
    /// \param p Pointer to destination for data
    /// \param n Number of bytes to read, must be integral number of elements.
    void read(void* p, size_t n) const override;

    TensorValue& get_value() { return m_tensor_value; }
    const TensorValue& get_value() const { return m_tensor_value; }
private:
    INTTensor(const INTTensor&) = delete;
    INTTensor(INTTensor&&) = delete;
    INTTensor& operator=(const INTTensor&) = delete;

    TensorValue m_tensor_value;
};
