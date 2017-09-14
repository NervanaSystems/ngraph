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

#include "ngraph/runtime/function.hpp"

namespace ngraph
{
    namespace runtime
    {
        class CallFrameAccessor;

        // This is constructed when a runtime function is called.
        class CallFrame
        {
            friend class CallFrameAccessor;

        public:
            CallFrame(Function&                             function,
                      const std::vector<std::shared_ptr<PrimaryTensorView>>& arguments,
                      const std::vector<std::shared_ptr<PrimaryTensorView>>& results);

        protected:
            std::vector<std::shared_ptr<PrimaryTensorView>> m_tensors;
        };

        class CallFrameAccessor
        {
        public:
            CallFrameAccessor(size_t index)
                : m_index(index)
            {
            }

            std::shared_ptr<PrimaryTensorView> operator()(CallFrame& call_frame)
            {
                return call_frame.m_tensors[m_index];
            }

        protected:
            size_t m_index;
        };
    }
}
