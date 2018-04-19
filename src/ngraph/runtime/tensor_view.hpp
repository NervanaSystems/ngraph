/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>

#include "ngraph/descriptor/tensor_view.hpp"
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
            TensorView(const std::shared_ptr<ngraph::descriptor::TensorView>& descriptor)
                : m_descriptor(descriptor)
            {
            }

        public:
            virtual ~TensorView() {}
            TensorView& operator=(const TensorView&) = default;

            std::shared_ptr<const ngraph::descriptor::TensorView>
                get_tensor_view_descriptor() const;

            virtual std::shared_ptr<descriptor::TensorView> get_descriptor() const;

            const ngraph::Shape& get_shape() const;
            const ngraph::Strides& get_strides() const;
            size_t get_element_count() const;
            const ngraph::descriptor::Tensor& get_tensor() const;

            std::shared_ptr<ngraph::descriptor::layout::TensorViewLayout>
                get_tensor_view_layout() const;

            /// @brief Write bytes directly into the tensor
            /// @param p Pointer to source of data
            /// @param tensor_offset Offset into tensor storage to begin writing. Must be element-aligned.
            /// @param n Number of bytes to write, must be integral number of elements.
            virtual void write(const void* p, size_t tensor_offset, size_t n) = 0;

            /// @brief Read bytes directly from the tensor
            /// @param p Pointer to destination for data
            /// @param tensor_offset Offset into tensor storage to begin reading. Must be element-aligned.
            /// @param n Number of bytes to read, must be integral number of elements.
            virtual void read(void* p, size_t tensor_offset, size_t n) const = 0;

        protected:
            std::shared_ptr<ngraph::descriptor::TensorView> m_descriptor;
        };

        using TensorViewPtrs = std::vector<std::shared_ptr<TensorView>>;
    }
}
