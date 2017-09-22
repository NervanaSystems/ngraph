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

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/types/type.hpp"
#include "ngraph/log.hpp"

namespace ngraph
{
    class Node;

    namespace descriptor
    {
        class Tensor;
        class TensorViewLayout;

        // Describes a view of an instantiated tensor
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
            virtual Tensor&       get_tensor()       = 0;
            const std::string& get_name() const { return m_name; }

            std::shared_ptr<const TensorViewType> get_tensor_view_type() const
            {
                return m_tensor_view_type;
            }

            const std::shared_ptr<TensorViewLayout>& get_tensor_view_layout() const
            {
                return m_tensor_view_layout;
            }

            void set_tensor_view_layout(const std::shared_ptr<TensorViewLayout>& tensor_view_layout)
            {
                m_tensor_view_layout = tensor_view_layout;
            }

        protected:
            std::shared_ptr<const TensorViewType> m_tensor_view_type;
            std::shared_ptr<TensorViewLayout>     m_tensor_view_layout;
            std::string                           m_name;
        };

        // A PrimaryTensorView owns the tensor. All other views are the result
        // of some index operation on the primary view.
        class PrimaryTensorView : public TensorView
        {
        public:
            PrimaryTensorView(const std::shared_ptr<const TensorViewType>& tensor_view_type,
                const Node* parent, size_t value_index);

            virtual const Tensor& get_tensor() const override;
            virtual Tensor&       get_tensor() override;

        protected:
            Tensor m_tensor;
        };
    }
}
