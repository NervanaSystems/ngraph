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
#include <vector>

#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    class TensorValue
    {
    public:
        TensorValue(TensorValue&) = delete;
        TensorValue(const TensorValue&) = delete;
        TensorValue& operator=(TensorValue&) = delete;
        TensorValue& operator=(const TensorValue&) = delete;
        TensorValue& operator=(TensorValue&& other)
        {
            m_element_type = other.m_element_type;
            m_shape = other.m_shape;
            m_raw_buffer = other.m_raw_buffer;
            other.m_element_type = element::f32;
            other.m_shape = Shape{0};
            other.m_raw_buffer = nullptr;
        }
        TensorValue()
            : TensorValue(element::f32, Shape{0}, nullptr)
        {
        }
        TensorValue(TensorValue&& other)
            : m_element_type(other.m_element_type)
            , m_shape(other.m_shape)
            , m_raw_buffer(other.m_raw_buffer)
        {
            other.m_element_type = element::f32;
            other.m_shape = Shape{0};
            other.m_raw_buffer = nullptr;
        }
        TensorValue(const element::Type& element_type, const Shape& shape, void* raw_buffer)
            : m_element_type(element_type)
            , m_shape(shape)
            , m_raw_buffer(raw_buffer)
        {
        }
        template <typename T>
        TensorValue(const Shape& shape, T* raw_buffer)
            : TensorValue(element::from<T>(), shape, raw_buffer)
        {
        }
        // template<typename T>
        // TensorValue(NDArrayBase<T>& nd_array)
        //     : TensorValue(element::from<T>(), nd_array.get_shape(), nd_array.data())
        // {
        // }
        const element::Type& element_type() const { return m_element_type; }
        const Shape& shape() const { return m_shape; }
        void* raw_buffer() { return m_raw_buffer; }
        const void* raw_buffer() const { return m_raw_buffer; }
        template <typename T>
        const T* buffer() const
        {
            NGRAPH_CHECK(m_element_type == element::from<T>());
            return reinterpret_cast<const T*>(m_raw_buffer);
        }
        template <typename T>
        T* buffer()
        {
            NGRAPH_CHECK(m_element_type == element::from<T>());
            return reinterpret_cast<T*>(m_raw_buffer);
        }

    private:
        element::Type m_element_type;
        Shape m_shape;
        void* m_raw_buffer;
    };
}
