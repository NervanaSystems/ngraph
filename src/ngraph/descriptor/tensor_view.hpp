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
#include <string>

#include "ngraph/shape.hpp"

namespace ngraph
{
    class Node;
    class TensorViewType;

    namespace descriptor
    {
        namespace layout
        {
            class Tensor;
            class TensorViewLayout;
        }

        class Tensor;
        class TensorView;

        /// @brief Compile-time descriptor of a first-class value that is a view of a tensor.
        class TensorView
        {
            TensorView(const TensorView&) = delete;
            TensorView& operator=(const TensorView&) = delete;

        protected:
            TensorView(const std::shared_ptr<const TensorViewType>& tensor_view_type)
                : m_tensor_view_type(tensor_view_type)
            {
            }

        public:
            virtual ~TensorView() {}
            virtual const Tensor& get_tensor() const = 0;
            virtual Tensor& get_tensor() = 0;

            virtual std::shared_ptr<const TensorViewType> get_value_type() const;

            const std::string& get_name() const { return m_name; }
            std::shared_ptr<const TensorViewType> get_tensor_view_type() const
            {
                return m_tensor_view_type;
            }

            const std::shared_ptr<layout::TensorViewLayout>& get_tensor_view_layout() const
            {
                return m_tensor_view_layout;
            }

            void set_tensor_view_layout(
                const std::shared_ptr<layout::TensorViewLayout>& tensor_view_layout)
            {
                m_tensor_view_layout = tensor_view_layout;
            }

        protected:
            std::shared_ptr<const TensorViewType> m_tensor_view_type;
            std::shared_ptr<layout::TensorViewLayout> m_tensor_view_layout;
            std::string m_name;
        };

        using TensorViewPtrs = std::vector<std::shared_ptr<TensorView>>;
    }
}
