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

#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class INT_Backend : public runtime::Backend
            {
            public:
                std::shared_ptr<ngraph::runtime::CallFrame> make_call_frame(
                    const std::shared_ptr<ngraph::runtime::ExternalFunction>& external_function)
                    override;

                std::shared_ptr<ngraph::runtime::TensorView>
                    make_primary_tensor_view(const ngraph::element::Type& element_type,
                                             const Shape& shape) override;

                std::shared_ptr<ngraph::runtime::TensorView>
                    make_primary_tensor_view(const ngraph::element::Type& element_type,
                                             const Shape& shape,
                                             void* memory_pointer) override;
            };
        }
    }
}
