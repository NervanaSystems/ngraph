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
#include "ngraph/function.hpp"

namespace ngraph
{
    namespace descriptor
    {
        // Describes the frame that will be used when a function is executing
        class CallFrame
        {
        protected:
            Function m_function;

            // Will be provided by the caller
            std::vector<std::shared_ptr<TensorView>> m_inputs;
            std::vector<std::shared_ptr<TensorView>> m_outputs;
            // Will be provided by the call mechanism
            // Expect there to be only one buffer
            std::vector<std::shared_ptr<Buffer>> m_buffers;
        };
    }
}
