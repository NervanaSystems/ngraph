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

namespace ngraph
{
    namespace element
    {
        class Type;
    }

    namespace descriptor
    {
        class TensorView;

        namespace layout
        {
            /// @brief Interface for describing implementations of tensor views.
            ///
            /// Kernel selection will need to pay attention to the layout.
            class TensorViewLayout
            {
            protected:
                TensorViewLayout(const ngraph::descriptor::TensorView& tensor_view);
                TensorViewLayout(const TensorViewLayout&) = delete;
                TensorViewLayout& operator=(const TensorViewLayout&) = delete;

            public:
                virtual ~TensorViewLayout() {}
                /// Extent of this view in buffer.
                ///
                /// When we support non-linear buffers, this will need to be something other than size_t.
                virtual size_t get_size() = 0;
                virtual size_t size();
                /// Offset of an index; useful for slice implementation.
                ///
                /// With non-linear buffers, this will need to be something other than size_t.
                virtual size_t get_index_offset(const std::vector<size_t>& indices) = 0;

                const element::Type& get_element_type() const;
                const Shape& get_shape() const;
                virtual const Strides& get_strides() const = 0;
                /// @brief Return true if this and other have the same element interpretation
                virtual bool operator==(const TensorViewLayout& other) const = 0;
                bool operator!=(const TensorViewLayout& other) const { return !(*this == other); }
            protected:
                std::shared_ptr<const TensorViewType> m_tensor_view_type;

            private:
                size_t m_size;
            };
        }
    }
}
