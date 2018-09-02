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
#include <string>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    class Node;

    namespace descriptor
    {
        namespace layout
        {
            class Tensor;
            class TensorViewLayout;
        }

        /// \brief Compile-time descriptor of a first-class value that is a view of a tensor.
        class TensorView
        {
            TensorView(const TensorView&) = delete;
            TensorView& operator=(const TensorView&) = delete;

        public:
            TensorView(const element::Type& element_type,
                       const Shape& shape,
                       const std::string& name);

            const Tensor& get_tensor() const;
            Tensor& get_tensor();

            const std::string& get_name() const { return m_name; }
            void set_tensor_view_type(const element::Type& element_type, const Shape& shape);

            const element::Type& get_element_type() const;
            const Shape& get_shape() const;

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
            element::Type m_element_type;
            Shape m_shape;
            std::shared_ptr<layout::TensorViewLayout> m_tensor_view_layout;
            std::string m_name;
            Tensor m_tensor;
        };

        using TensorViewPtrs = std::vector<std::shared_ptr<TensorView>>;
    }
}
