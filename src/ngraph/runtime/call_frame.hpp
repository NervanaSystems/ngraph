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

#include "ngraph/function.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        class Value;

        // A VM for executing lightly-compiled graph functions.
        class CallFrame
        {
        public:
            virtual ~CallFrame() {}
            /// @brief Invoke the function with values matching the signature of the function.
            ///
            /// Tuples will be expanded into their tensor views to build the call frame.
            virtual void
                call(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& inputs,
                     const std::vector<std::shared_ptr<ngraph::runtime::Value>>& outputs) = 0;

            /// @brief Invoke the function with tuples pre-expanded to their underlying tensor views.
            virtual void tensor_call(const TensorViewPtrs& inputs,
                                     const TensorViewPtrs& outputs) = 0;
        };
    }
}
