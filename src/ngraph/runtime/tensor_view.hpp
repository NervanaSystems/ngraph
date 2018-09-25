//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/descriptor/tensor.hpp"
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
        class TensorView
        {
        protected:
            TensorView(const std::shared_ptr<ngraph::descriptor::Tensor>& descriptor)
                : m_descriptor(descriptor)
                , m_stale(true)
            {
            }

        public:
            virtual ~TensorView() {}
            TensorView& operator=(const TensorView&) = default;

            const ngraph::Shape& get_shape() const;
            ngraph::Strides get_strides() const;
            const element::Type& get_element_type() const;
            size_t get_size() const;
            const std::string& get_name() const;
            std::shared_ptr<descriptor::layout::TensorLayout> get_tensor_layout() const;
            void set_tensor_layout(const std::shared_ptr<descriptor::layout::TensorLayout>& layout);

            bool get_stale() const { return m_stale; }
            void set_stale(bool val) { m_stale = val; }
            /// \brief Write bytes directly into the tensor
            /// \param p Pointer to source of data
            /// \param tensor_offset Offset into tensor storage to begin writing. Must be element-aligned.
            /// \param n Number of bytes to write, must be integral number of elements.
            virtual void write(const void* p, size_t tensor_offset, size_t n) = 0;

            /// \brief Read bytes directly from the tensor
            /// \param p Pointer to destination for data
            /// \param tensor_offset Offset into tensor storage to begin reading. Must be element-aligned.
            /// \param n Number of bytes to read, must be integral number of elements.
            virtual void read(void* p, size_t tensor_offset, size_t n) const = 0;

        protected:
            std::shared_ptr<ngraph::descriptor::Tensor> m_descriptor;
            bool m_stale;
        };

        using TensorViewPtrs = std::vector<std::shared_ptr<TensorView>>;
    }
}
