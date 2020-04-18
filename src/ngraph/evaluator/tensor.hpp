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

#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type_traits.hpp"

namespace ngraph
{
    namespace evaluator
    {
        class Tensor
        {
        public:
            ngraph::element::Type_t get_element_type() const { return m_element_type; };
            const ngraph::Shape& get_shape() { return m_shape; }
            size_t size() { return m_size; }
            size_t get_byte_size() { return m_size * ngraph::compiler_byte_size(m_element_type); }
            /// \brief Tensor that uses data for its memory
            Tensor(ngraph::element::Type_t element_type,
                   const ngraph::Shape& shape,
                   const void* read_data)
                : m_element_type(element_type)
                , m_shape(shape)
                , m_size(shape_size(shape))
                , m_read_data(read_data)
            {
            }

            Tensor(const std::shared_ptr<ngraph::op::Constant>& c);

            /// \brief Tensor that owns its memory
            Tensor(ngraph::element::Type_t element_type, const ngraph::Shape& shape)
                : m_element_type(element_type)
                , m_shape(shape)
                , m_size(shape_size(shape))
                , m_aligned(new ngraph::runtime::AlignedBuffer(get_byte_size()))
                , m_write_data(m_aligned->get_ptr())
                , m_read_data(m_write_data)
            {
            }

            Tensor()
                : m_element_type(ngraph::element::Type_t::undefined)
                , m_shape({})
                , m_size(0)
            {
            }

            template <ngraph::element::Type_t ET>
            const typename ngraph::element_type_traits<ET>::value_type* get_read_data() const
            {
                if (ET != m_element_type)
                {
                    throw std::invalid_argument("Element type does not match C type");
                }
                if (m_read_data == nullptr)
                {
                    throw std::invalid_argument("Attempt to read invalid tensor");
                }
                return static_cast<const typename ngraph::element_type_traits<ET>::value_type*>(
                    m_read_data);
            }

            template <ngraph::element::Type_t ET>
            typename ngraph::element_type_traits<ET>::value_type* get_write_data()
            {
                if (ET != m_element_type)
                {
                    throw std::invalid_argument("Element type does not match C type");
                }
                if (m_write_data == nullptr)
                {
                    throw std::invalid_argument("Attempt to write invalid tensor");
                }
                return static_cast<typename ngraph::element_type_traits<ET>::value_type*>(
                    m_write_data);
            }

            template <ngraph::element::Type_t ET>
            Tensor& set_elements(
                const std::vector<typename ngraph::element_type_traits<ET>::value_type>& data)
            {
                if (data.size() > size())
                {
                    throw std::invalid_argument("Setting too many elements");
                }
                typename ngraph::element_type_traits<ET>::value_type* pdata = get_write_data<ET>();
                for (auto value : data)
                {
                    *(pdata++) = value;
                }
                return *this;
            }

        private:
            ngraph::element::Type_t m_element_type;
            ngraph::Shape m_shape;
            size_t m_size;
            std::shared_ptr<ngraph::runtime::AlignedBuffer> m_aligned;
            void* m_write_data{nullptr};
            const void* m_read_data{nullptr};
        };
    }
}
