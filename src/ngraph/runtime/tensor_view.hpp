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

#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/runtime/value.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/types/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        template <typename ET>
        class ParameterizedTensorView;

        class TensorView : public Value
        {
        protected:
            TensorView(const std::shared_ptr<ngraph::descriptor::TensorView>& descriptor)
                : m_descriptor(descriptor)
            {
            }

        public:
            TensorView() {}

            virtual ~TensorView() {}

            template <typename ET>
            ParameterizedTensorView<ET>* get_parameterized_tensor()
            {
                return dynamic_cast<ParameterizedTensorView<ET>*>(this);
            }

            std::shared_ptr<const ngraph::descriptor::TensorView> get_tensor_view_descriptor() const
            {
                return m_descriptor;
            }

            virtual std::shared_ptr<ngraph::descriptor::Value> get_descriptor() const override
            {
                return m_descriptor;
            }

            virtual void collect_tensor_views(std::vector<std::shared_ptr<TensorView>>& views,
                                              const std::shared_ptr<Value>& value) const override
            {
                views.push_back(std::static_pointer_cast<TensorView>(value));
            }

            const Shape& get_shape() { return m_descriptor->get_tensor_view_type()->get_shape(); }

        protected:
            std::shared_ptr<ngraph::descriptor::TensorView> m_descriptor;
        };
    }
}
