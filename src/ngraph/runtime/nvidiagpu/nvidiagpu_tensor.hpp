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
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nvidiagpu
        {
            class Tensor;
        }
    }
}

class ngraph::runtime::nvidiagpu::Tensor : public ngraph::runtime::Tensor
{
public:
    Tensor(const ngraph::element::Type& element_type, const ngraph::Shape& shape, const ngraph::runtime::Backend* parent);
    Tensor(const ngraph::element::Type& element_type,
           const ngraph::Shape& shape,
             void* memory_pointer,
           const ngraph::runtime::Backend* parent);
    virtual ~Tensor() override;

    /// \brief Write bytes directly into the tensor
    /// \param p Pointer to source of data
    /// \param tensor_offset Offset into tensor storage to begin writing. Must be element-aligned.
    /// \param n Number of bytes to write, must be integral number of elements.
    void write(const void* p, size_t tensor_offset, size_t n) override;

    /// \brief Read bytes directly from the tensor
    /// \param p Pointer to destination for data
    /// \param tensor_offset Offset into tensor storage to begin reading. Must be element-aligned.
    /// \param n Number of bytes to read, must be integral number of elements.
    void read(void* p, size_t tensor_offset, size_t n) const override;

    void* m_allocated_buffer_pool = nullptr;
    size_t m_buffer_size;
    bool m_custom_memory;

private:
    Tensor(const ngraph::runtime::Tensor&) = delete;
    Tensor(ngraph::runtime::Tensor&&) = delete;
    Tensor& operator=(const ngraph::runtime::Tensor&) = delete;
};
