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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        template <typename ET>
        class ParameterizedTensorView : public runtime::TensorView
        {
        public:
            /// Create a tensor
            ParameterizedTensorView(const ngraph::Shape& shape)
                : TensorView(std::make_shared<ngraph::descriptor::PrimaryTensorView>(
                      std::make_shared<ngraph::TensorViewType>(ET::element_type(), shape),
                      "external",
                      true,
                      true))
            {
                m_descriptor->set_tensor_view_layout(
                    std::make_shared<ngraph::descriptor::layout::DenseTensorViewLayout>(
                        *m_descriptor));
                m_vector.resize(m_descriptor->get_tensor_view_layout()->get_size());
            }

            using value_type = typename ET::type;
            using storage_type = std::vector<value_type>;

            // For getting the data out
            const storage_type& get_vector() const { return m_vector; }
            void* get_data_ptr() { return m_vector.data(); }
            virtual void write(const void* p, size_t tensor_offset, size_t n) override
            {
                size_t elt_offset = tensor_offset / sizeof(typename ET::type);
                if (elt_offset * sizeof(typename ET::type) != tensor_offset)
                {
                    throw ngraph_error("Attempt to write to an address not aligned on an element");
                }
                size_t elt_n = n / sizeof(typename ET::type);
                if (elt_n * sizeof(typename ET::type) != n)
                {
                    throw ngraph_error("Attemmpt to write a partial element");
                }
                size_t elt_byte_size = sizeof(typename ET::type) * n;
                if (tensor_offset + n > elt_byte_size)
                {
                    throw ngraph_error("Attempt to write beyond the tensor");
                }

                std::memcpy(&m_vector[elt_offset], p, n);
            }

            template <typename T>
            void write(const std::vector<T>& values)
            {
                write(values.data(), 0, values.size() * sizeof(T));
            }

            virtual void read(void* p, size_t tensor_offset, size_t n) const override
            {
                size_t elt_offset = tensor_offset / sizeof(typename ET::type);
                if (elt_offset * sizeof(typename ET::type) != tensor_offset)
                {
                    throw ngraph_error("Attempt to read from an address not aligned on an element");
                }
                size_t elt_n = n / sizeof(typename ET::type);
                if (elt_n * sizeof(typename ET::type) != n)
                {
                    throw ngraph_error("Attemmpt to read a partial element");
                }
                size_t elt_byte_size = sizeof(typename ET::type) * n;
                if (tensor_offset + n > elt_byte_size)
                {
                    throw ngraph_error("Attempt to read beyond the tensor");
                }

                std::memcpy(p, &m_vector[elt_offset], n);
            }

        protected:
            storage_type m_vector;
        };
    }
}
