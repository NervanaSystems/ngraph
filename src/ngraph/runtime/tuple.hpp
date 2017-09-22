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

#include "ngraph/descriptor/tuple.hpp"
#include "ngraph/runtime/value.hpp"

namespace ngraph
{
    namespace runtime
    {
        /// @brief A first-class value holding zero or more first-class values.
        class Tuple : public Value
        {
        public:
            Tuple(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& elements);

            virtual std::shared_ptr<ngraph::descriptor::Value> get_descriptor() const override
            {
                return m_descriptor;
            }

            std::shared_ptr<const ngraph::descriptor::Value> get_tuple_descriptor() const
            {
                return m_descriptor;
            }

            virtual void
                collect_tensor_views(std::vector<std::shared_ptr<TensorView>>& views,
                                     const std::shared_ptr<Value>& value) const override;

        protected:
            std::vector<std::shared_ptr<Value>>        m_elements;
            std::shared_ptr<ngraph::descriptor::Tuple> m_descriptor;
        };
    }
}
