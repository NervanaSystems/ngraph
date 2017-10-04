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

#include "ngraph/descriptor/value.hpp"
#include "ngraph/types/type.hpp"

namespace ngraph
{
    namespace descriptor
    {
        /// @brief Compile-time descriptor of a first-class value that is a tuple of zero or more first-class values.
        class Tuple : public Value
        {
        public:
            Tuple(const std::vector<std::shared_ptr<ngraph::descriptor::Value>>& elements);

            const std::shared_ptr<ngraph::TupleType> get_tuple_type() const;
            std::shared_ptr<ngraph::TupleType> get_tuple_type();

            virtual std::shared_ptr<const ValueType> get_value_type() const override
            {
                return m_tuple_type;
            }

            virtual void collect_tensor_views(std::vector<std::shared_ptr<TensorView>>& views,
                                              const std::shared_ptr<Value>& value) const override;

        protected:
            std::shared_ptr<ngraph::TupleType> m_tuple_type;
            std::vector<std::shared_ptr<ngraph::descriptor::Value>> m_elements;
        };
    }
}
