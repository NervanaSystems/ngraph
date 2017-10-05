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
#include <vector>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        template <typename ET>
        class ParameterizedTensorView : public TensorView
        {
        public:
            /// Create a tensor
            ParameterizedTensorView(const ngraph::Shape& shape)
                : TensorView(std::make_shared<ngraph::descriptor::PrimaryTensorView>(
                      std::make_shared<ngraph::TensorViewType>(ET::element_type(), shape),
                      "external",
                      true,
                      true))
            {
                m_descriptor->set_tensor_view_layout(
                    std::make_shared<ngraph::descriptor::layout::DenseTensorViewLayout>(
                        m_descriptor));
                m_vector.resize(m_descriptor->get_tensor_view_layout()->get_size());
            }

            ParameterizedTensorView(
                const std::shared_ptr<ngraph::descriptor::TensorView>& descriptor);

            using element_type = ET;
            using value_type = typename ET::type;
            using storage_type = std::vector<value_type>;

            template <typename T>
            ParameterizedTensorView<ET>& operator=(const std::vector<T>& value)
            {
                get_vector() = value;
                return *this;
            }

            // For getting the data out
            storage_type& get_vector() { return m_vector; }
        protected:
            storage_type m_vector;
        };
    }
}
