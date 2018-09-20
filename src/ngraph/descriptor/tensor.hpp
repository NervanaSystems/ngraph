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
            class TensorLayout;
        }

        /// \brief Compile-time descriptor of a first-class value that is a view of a tensor.
        class Tensor
        {
            Tensor(const Tensor&) = delete;
            Tensor& operator=(const Tensor&) = delete;

        public:
            Tensor(const element::Type& element_type, const Shape& shape, const std::string& name);

            const std::string& get_name() const { return m_name; }
            void set_tensor_view_type(const element::Type& element_type, const Shape& shape);

            const element::Type& get_element_type() const { return m_element_type; }
            const Shape& get_shape() const { return m_shape; }
            const std::shared_ptr<layout::TensorLayout>& get_tensor_layout() const
            {
                return m_tensor_layout;
            }

            void set_tensor_layout(const std::shared_ptr<layout::TensorLayout>& tensor_layout);

            void set_pool_offset(size_t);
            size_t get_pool_offset() const;

            size_t size() const;

        protected:
            element::Type m_element_type;
            Shape m_shape;
            std::string m_name;
            std::shared_ptr<layout::TensorLayout> m_tensor_layout;
            size_t m_pool_offset{0};
        };

        using TensorView = Tensor;

        using TensorViewPtrs = std::vector<std::shared_ptr<Tensor>>;
        std::ostream& operator<<(std::ostream&, const ngraph::descriptor::Tensor&);
    }
}
