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

#include <tuple>
#include <vector>

#include "ngraph/descriptor/buffer_pos.hpp"
#include "ngraph/descriptor/tensor_view.hpp"

namespace ngraph
{
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
                TensorViewLayout(const ngraph::descriptor::TensorView& tensor_view)
                    : m_tensor_view(tensor_view)
                {
                }

            public:
                virtual ~TensorViewLayout() {}
                /// Extent of this view in buffer.
                ///
                /// When we support non-linear buffers, this will need to be something other than size_t.
                virtual size_t get_size() = 0;

                /// Offset of an index; useful for slice implementation.
                ///
                /// With non-linear buffers, this will need to be something other than size_t.
                virtual size_t get_index_offset(const std::vector<size_t>& indices) = 0;

                const Shape& get_shape() const
                {
                    return m_tensor_view.get_tensor_view_type()->get_shape();
                }

                /// Where this view is located in the buffer.
                const BufferPos& get_buffer_pos() const { return m_buffer_pos; }
                BufferPos& get_buffer_pos() { return m_buffer_pos; }
            protected:
                const ngraph::descriptor::TensorView& m_tensor_view;
                BufferPos m_buffer_pos;
            };
        }
    }
}
