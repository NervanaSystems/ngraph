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
    class ValueType;

    namespace descriptor
    {
        class TensorView;

        /// @brief Compile-time descriptor of a first-class value.
        class Value
        {
        public:
            virtual ~Value() {}
            virtual std::shared_ptr<const ngraph::ValueType> get_value_type() const = 0;

            /// @brief helper for collecting all the tensor views in a sequence of values
            ///
            /// @param views The vector of tensor views being collected.
            /// @param value A shared pointer for this.
            ///
            /// Append each tensor view in this value to views. Since this may be a tensor view
            /// we need to pass a shared pointer to this since we can't get one from this.
            virtual void collect_tensor_views(std::vector<std::shared_ptr<TensorView>>& views,
                                              const std::shared_ptr<Value>& value) const = 0;
        };
    }
}
