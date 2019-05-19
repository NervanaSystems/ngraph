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

#include <future>
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
            friend class Executable;

        protected:
            Tensor(const std::shared_ptr<ngraph::descriptor::Tensor>& descriptor)
                : m_descriptor(descriptor)
                , m_stale(true)
            {
            }

            Tensor(const std::shared_ptr<ngraph::runtime::Backend>& backend,
                   const std::shared_ptr<ngraph::descriptor::Tensor>& descriptor)
                : m_descriptor(descriptor)
                , m_stale(true)
                , m_backend{backend}
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
            /// \param offset Offset into tensor storage to begin writing. Must be element-aligned.
            /// \param n Number of bytes to write, must be integral number of elements.
            virtual void write(const void* p, size_t offset, size_t n) = 0;

            /// \brief Read bytes directly from the tensor
            /// \param p Pointer to destination for data
            /// \param offset Offset into tensor storage to begin writing. Must be element-aligned.
            /// \param n Number of bytes to read, must be integral number of elements.
            virtual void read(void* p, size_t offset, size_t n) const = 0;

            /// \brief Write bytes into the tensor. The data buffer pointed to by `p` must
            ///     be kept live until after the future is signaled complete
            /// \param p Pointer to source of data
            /// \param n Number of bytes to write, must be integral number of elements.
            /// \return std::future to track the operation
            virtual std::future<void> begin_write(const void* p, size_t n);

            /// \brief Read bytes from the tensor. The data buffer pointed to by `p` must
            ///     be kept live until after the future is signaled complete
            /// \param p Pointer to destination for data
            /// \param n Number of bytes to read, must be integral number of elements.
            /// \return std::future to track the operation
            virtual std::future<void> begin_read(void* p, size_t n);

            /// \brief copy bytes directly from source to this tensor
            /// \param source The source tensor
            virtual void copy_from(const ngraph::runtime::Tensor& source);

        protected:
            std::shared_ptr<ngraph::descriptor::Tensor> m_descriptor;
            bool m_stale;
            std::promise<void> m_promise;
            std::shared_ptr<ngraph::runtime::Backend> m_backend;
        };
    }
}
