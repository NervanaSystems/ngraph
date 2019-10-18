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

#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace descriptor
    {
        class Value;
    }

    namespace runtime
    {
        class Tensor
        {
        protected:
            Tensor(const std::shared_ptr<ngraph::descriptor::Tensor>& descriptor)
                : m_descriptor(descriptor)
                , m_stale(true)
            {
            }

        public:
            virtual ~Tensor() {}
            Tensor& operator=(const Tensor&) = default;

            /// \brief Get tensor shape
            /// \return const reference to a Shape
            virtual const ngraph::Shape& get_shape() const;

            /// \brief Get tensor partial shape
            /// \return const reference to a PartialShape
            const ngraph::PartialShape& get_partial_shape() const;

            /// \brief Get tensor strides
            /// \return Strides
            virtual ngraph::Strides get_strides() const;

            /// \brief Get tensor element type
            /// \return element::Type
            virtual const element::Type& get_element_type() const;

            /// \brief Get number of elements in the tensor
            /// \return number of elements in the tensor
            virtual size_t get_element_count() const;

            /// \brief Get the size in bytes of the tensor
            /// \return number of bytes in tensor's allocation
            virtual size_t get_size_in_bytes() const;

            /// \brief Get tensor's unique name
            /// \return tensor's name
            const std::string& get_name() const;

            /// \brief Get tensor layout
            /// \return tensor layout
            std::shared_ptr<descriptor::layout::TensorLayout> get_tensor_layout() const;

            /// \brief Set tensor layout
            /// \param layout Layout to set
            void set_tensor_layout(const std::shared_ptr<descriptor::layout::TensorLayout>& layout);

            /// \brief Get the stale value of the tensor. A tensor is stale if its data is
            /// changed.
            /// \return true if there is new data in this tensor
            bool get_stale() const;

            /// \brief Set the stale value of the tensor. A tensor is stale if its data is
            /// changed.
            void set_stale(bool val);

            /// \brief Write bytes directly into the tensor
            /// \param p Pointer to source of data
            /// \param n Number of bytes to write, must be integral number of elements.
            virtual void write(const void* p, size_t n) = 0;

            /// \brief Read bytes directly from the tensor
            /// \param p Pointer to destination for data
            /// \param n Number of bytes to read, must be integral number of elements.
            virtual void read(void* p, size_t n) const = 0;

            /// \brief copy bytes directly from source to this tensor
            /// \param source The source tensor
            virtual void copy_from(const ngraph::runtime::Tensor& source);

        protected:
            std::shared_ptr<ngraph::descriptor::Tensor> m_descriptor;
            bool m_stale;
        };
    }
}

#include "paranoid_vector.h"
namespace std {
    template<>
        class vector<shared_ptr<ngraph::runtime::Tensor>, allocator<shared_ptr<ngraph::runtime::Tensor> > >
        : public paranoid_vector<shared_ptr<ngraph::runtime::Tensor>> {
            public:
                using base_class = paranoid_vector<shared_ptr<ngraph::runtime::Tensor>>;
                using value_type = base_class::value_type;

                using allocator_type         = base_class::allocator_type;
                using size_type              = base_class::size_type;
                using difference_type        = base_class::difference_type;
                using reference              = base_class::reference;
                using const_reference        = base_class::const_reference;
                using pointer                = base_class::pointer;
                using const_pointer          = base_class::const_pointer;
                using iterator               = base_class::iterator;
                using const_iterator         = base_class::const_iterator;
                using reverse_iterator       = base_class::reverse_iterator;
                using const_reverse_iterator = base_class::const_reverse_iterator;

                vector(size_t pool_preferred_max_size_bytes)                                           : base_class(pool_preferred_max_size_bytes) {}
                vector(std::shared_ptr<allocator_type> allocator = std::make_shared<allocator_type>()) : base_class(allocator) {}
                vector(const vector<value_type>& other)                                            : base_class(other) {}
                vector(std::initializer_list<value_type> l)                                            : base_class(l) {}
                vector& operator=( const vector& other ) { base_class::operator=(other); return *this; }

                int foozle;
        };
}

namespace ngraph {
    namespace runtime {
        using TensorViewPtrs = std::vector<std::shared_ptr<Tensor>>;
    }
}

