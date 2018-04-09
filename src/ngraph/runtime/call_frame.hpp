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
#include <vector>

#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        // A VM for executing lightly-compiled graph functions.
        class CallFrame
        {
        public:
            virtual ~CallFrame() {}
            /// @brief Invoke the function with values matching the signature of the function.
            ///
            virtual void call(const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
                              const std::vector<std::shared_ptr<runtime::TensorView>>& inputs) = 0;

            /// @brief Invoke the function
            virtual void tensor_call(const TensorViewPtrs& outputs,
                                     const TensorViewPtrs& inputs) = 0;

            virtual std::vector<PerformanceCounter> get_performance_data() const
            {
                return std::vector<PerformanceCounter>();
            }
        };
    }
}
