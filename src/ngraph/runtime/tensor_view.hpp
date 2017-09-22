// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "ngraph/shape.hpp"
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        template <typename ET>
        class ParameterizedTensorView;

        class TensorView
        {
        public:
            virtual ~TensorView() {}
            template <typename ET>
            ParameterizedTensorView<ET>* get_parameterized_tensor()
            {
                return dynamic_cast<ParameterizedTensorView<ET>*>(this);
            }
        };

        template <typename ET>
        class ParameterizedTensorView : public ngraph::runtime::TensorView
        {
        public:
            ParameterizedTensorView(const ngraph::Shape& shape)
                : m_vector(ngraph::shape_size(shape), 0)
                , m_shape(shape)
            {
            }

            virtual ~ParameterizedTensorView() {}

            // Standard definitions from vector
            using element_type           = ET;
            using value_type             = typename ET::type;
            using storage_type           = std::vector<value_type>;
            using size_type              = typename storage_type::size_type;
            using difference_type        = typename storage_type::difference_type;
            using reference              = typename storage_type::reference;
            using const_reference        = typename storage_type::const_reference;
            using pointer                = typename storage_type::pointer;
            using const_pointer          = typename storage_type::const_pointer;
            using iterator               = typename storage_type::iterator;
            using const_iterator         = typename storage_type::const_iterator;
            using reverse_iterator       = typename storage_type::reverse_iterator;
            using const_reverse_iterator = typename storage_type::const_reverse_iterator;

            template <typename T>
            ParameterizedTensorView<ET>& operator=(const std::vector<T>& value)
            {
                get_vector() = value;
                return *this;
            }

            // For getting the data out
            storage_type& get_vector() { return m_vector; }
            const ngraph::Shape& get_shape() const { return m_shape; }

        protected:
            storage_type m_vector;
            ngraph::Shape m_shape;
        };
    }
}
