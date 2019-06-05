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
        class HybridTensor;
    }
}

class ngraph::runtime::HybridTensor : public ngraph::runtime::Tensor
{
public:
    HybridTensor(const ngraph::element::Type& element_type, const Shape& shape);
    HybridTensor(const ngraph::element::Type& element_type,
                 const Shape& shape,
                 void* memory_pointer);
    virtual ~HybridTensor() override;

    char* get_data_ptr();
    const char* get_data_ptr() const;

    template <typename T>
    T* get_data_ptr()
    {
        return reinterpret_cast<T*>(get_data_ptr());
    }

    template <typename T>
    const T* get_data_ptr() const
    {
        return reinterpret_cast<T*>(get_data_ptr());
    }

    /// \brief Write bytes directly into the tensor
    /// \param p Pointer to source of data
    /// \param n Number of bytes to write, must be integral number of elements.
    void write(const void* p, size_t n) override;

    /// \brief Read bytes directly from the tensor
    /// \param p Pointer to destination for data
    /// \param n Number of bytes to read, must be integral number of elements.
    void read(void* p, size_t n) const override;

protected:
    HybridTensor(const HybridTensor&) = delete;
    HybridTensor(HybridTensor&&) = delete;
    HybridTensor& operator=(const HybridTensor&) = delete;

    char* m_allocated_buffer_pool;
    char* m_aligned_buffer_pool;
    size_t m_buffer_size;
};
