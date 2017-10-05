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

namespace ngraph
{
    namespace runtime
    {
        /// @brief Compile-time information about a tensor view.
        ///
        /// Contains the offset of the tensor view in the call frame and the tensor descriptor.
        class TensorViewInfo
        {
        public:
            TensorViewInfo(size_t index,
                           const std::shared_ptr<ngraph::descriptor::TensorView>& descriptor)
                : m_index(index)
                , m_layout(descriptor->get_tensor_view_layout())
            {
            }

            size_t get_index() const { return m_index; }
            std::shared_ptr<ngraph::descriptor::layout::TensorViewLayout>
                get_tensor_view_layout() const
            {
                return m_layout;
            }

            template <typename LT>
            std::shared_ptr<LT> get_layout() const
            {
                return std::static_pointer_cast<LT>(m_layout);
            }

        protected:
            size_t m_index;
            std::shared_ptr<ngraph::descriptor::layout::TensorViewLayout> m_layout;
        };
    }
}
