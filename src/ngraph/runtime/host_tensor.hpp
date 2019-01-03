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
        static size_t alignment = 64;

        class HostTensor;
    }
}

class ngraph::runtime::HostTensor : public ngraph::runtime::Tensor
{
public:
    HostTensor(const ngraph::element::Type& element_type,
               const Shape& shape,
               const std::string& name = "external",
               const Backend* parent = nullptr);
    HostTensor(const ngraph::element::Type& element_type,
               const Shape& shape,
               void* memory_pointer,
               const std::string& name = "external",
               const Backend* parent = nullptr);
    HostTensor(const ngraph::element::Type& element_type,
               const Shape& shape,
               const Backend* parent);
    HostTensor(const ngraph::element::Type& element_type,
               const Shape& shape,
               void* memory_pointer,
               const Backend* parent);
    virtual ~HostTensor() override;

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
    /// \param tensor_offset Offset into tensor storage to begin writing. Must be element-aligned.
    /// \param n Number of bytes to write, must be integral number of elements.
    void write(const void* p, size_t tensor_offset, size_t n) override;

    /// \brief Read bytes directly from the tensor
    /// \param p Pointer to destination for data
    /// \param tensor_offset Offset into tensor storage to begin reading. Must be element-aligned.
    /// \param n Number of bytes to read, must be integral number of elements.
    void read(void* p, size_t tensor_offset, size_t n) const override;

private:
    HostTensor(const HostTensor&) = delete;
    HostTensor(HostTensor&&) = delete;
    HostTensor& operator=(const HostTensor&) = delete;

    char* m_allocated_buffer_pool;
    char* m_aligned_buffer_pool;
    size_t m_buffer_size;
};
