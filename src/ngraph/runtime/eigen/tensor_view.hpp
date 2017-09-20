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

#include <Eigen/Dense>
#include <vector>

#include "ngraph/shape.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/descriptor/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace eigen
        {
            std::shared_ptr<ngraph::runtime::PrimaryTensorView> make_tensor_view(std::shared_ptr<ngraph::descriptor::TensorView>);

            template <typename ET>
            class PrimaryTensorView : public ngraph::runtime::PrimaryTensorView
            {
            public:
                // Standard definitions from vector
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

                // Mapping vector to eigen
                using eigen_type = Eigen::Array<value_type, Eigen::Dynamic, 1>;
                using eigen_map  = Eigen::Map<eigen_type>;

                PrimaryTensorView(const ngraph::Shape& shape)
                    : m_shape(shape)
                    , m_size(ngraph::shape_size(shape))
                    , m_strides(ngraph::row_major_strides(m_shape))
                    , m_vector(m_size, 0)
                    , m_map(&m_vector[0], m_size, 1)
                {
                }

                template <typename T>
                PrimaryTensorView& operator=(const T& value)
                {
                    m_vector = value;
                    return *this;
                }

                // For getting the data out
                const storage_type& get_vector() { return m_vector; }

                eigen_map&       get_map() { return m_map; }
                const eigen_map& get_map() const { return m_map; }

                const Shape& get_shape() const { return m_shape; }

            protected:
                ngraph::Shape   m_shape;
                size_t          m_size;
                ngraph::Strides m_strides;
                storage_type    m_vector;
                eigen_map       m_map;
            };

            template <typename ET>
            void add(const PrimaryTensorView<ET>& arg0,
                     const PrimaryTensorView<ET>& arg1,
                     PrimaryTensorView<ET>&       out)
            {
                out.get_map() = arg0.get_map() + arg1.get_map();
            }

            template <typename ET>
            void multiply(const PrimaryTensorView<ET>& arg0,
                          const PrimaryTensorView<ET>& arg1,
                          PrimaryTensorView<ET>&       out)
            {
                out.get_map() = arg0.get_map() * arg1.get_map();
            }
        }
    }
}
