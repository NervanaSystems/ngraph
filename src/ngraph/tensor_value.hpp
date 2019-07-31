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

#include <cstring>
#include <memory>
#include <vector>

// TODO(amprocte): See about pulling `aligned_buffer` and `allocator` out of
// `runtime`.
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/allocator.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    class TensorBase
    {
    public:
        ~TensorBase() {}
        const element::Type& element_type() const { return m_element_type; }
        const Shape& shape() const { return m_shape; }
        template <typename T>
        const T* buffer() const
        {
            NGRAPH_CHECK(m_element_type == element::from<T>());
            return reinterpret_cast<const T*>(raw_buffer());
        }
        template <typename T>
        T* buffer()
        {
            NGRAPH_CHECK(m_element_type == element::from<T>());
            return reinterpret_cast<T*>(raw_buffer());
        }

        template <typename T>
        const T* buffer_unsafe() const
        {
            return reinterpret_cast<const T*>(raw_buffer());
        }
        template <typename T>
        T* buffer_unsafe()
        {
            return reinterpret_cast<T*>(raw_buffer());
        }

        virtual void* raw_buffer() = 0;
        virtual const void* raw_buffer() const = 0;

    protected:
        TensorBase(const element::Type& element_type, const Shape& shape)
            : m_element_type(element_type)
            , m_shape(shape)
        {
        }

        // TODO: would like to make this private, but TensorValue's move ctor
        // needs to zero it out.
        Shape m_shape;

    private:
        element::Type m_element_type;
    };

    class TensorMap : public TensorBase
    {
    public:
        TensorMap(const element::Type& element_type, const Shape& shape, void* raw_buffer)
            : TensorBase(element_type, shape)
            , m_raw_buffer(raw_buffer)
        {
        }
        TensorMap()
            : TensorMap(element::f32, Shape{0}, nullptr)
        {
        }
        TensorMap(const TensorMap& other)
            : TensorMap(other.element_type(), other.shape(), other.m_raw_buffer)
        {
        }
        template <typename T>
        TensorMap(const Shape& shape, T* raw_buffer)
            : TensorMap(element::from<T>(), shape, raw_buffer)
        {
        }

        void* raw_buffer() final override { return m_raw_buffer; }
        const void* raw_buffer() const final override { return m_raw_buffer; }
    private:
        void* m_raw_buffer;
    };

    class TensorValue : public TensorBase
    {
    public:
        TensorValue(const element::Type& element_type,
                    const Shape& shape,
                    void* raw_buffer,
                    runtime::Allocator* allocator = nullptr)
            : TensorBase(element_type, shape)
            , m_aligned_buffer(
                  shape_size(shape) * element_type.size(), element_type.size(), allocator)
        {
            NGRAPH_CHECK(raw_buffer != nullptr || shape_size(shape) == 0);
            if (raw_buffer != nullptr)
            {
                std::memcpy(m_aligned_buffer.get_ptr(),
                            raw_buffer,
                            shape_size(shape) * element_type.size());
            }
        }
        TensorValue()
            : TensorValue(element::f32, Shape{0}, nullptr, nullptr)
        {
        }
        TensorValue(runtime::Allocator* allocator)
            : TensorValue(element::f32, Shape{0}, nullptr, allocator)
        {
        }
        TensorValue(TensorValue&& other)
            : TensorBase(other.element_type(), other.shape())
            , m_aligned_buffer(std::move(other.m_aligned_buffer))
        {
            other.m_shape = Shape{0};
        }
        template <typename T>
        TensorValue(const Shape& shape, T* raw_buffer, runtime::Allocator* allocator = nullptr)
            : TensorValue(element::from<T>(), shape, raw_buffer, allocator)
        {
        }
        template <typename T>
        TensorValue(const Shape& shape,
                    const std::vector<T>& vec,
                    runtime::Allocator* allocator = nullptr)
            : TensorValue(element::from<T>(), shape, vec.data(), allocator)
        {
        }

        void* raw_buffer() final override { return m_aligned_buffer.get_ptr(); }
        const void* raw_buffer() const final override { return m_aligned_buffer.get_ptr(); }
    private:
        runtime::AlignedBuffer m_aligned_buffer;
    };
}
