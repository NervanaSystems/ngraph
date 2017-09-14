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

#include <vector>

namespace ngraph
{
    namespace descriptor
    {
        // An interface for describing implementations of tensor views
        // Kernel selection will need to pay attention to the layout
        class TensorViewLayout
        {
        public:
            virtual ~TensorViewLayout() {}
        };

        // The standard strided layout
        class DenseTensorViewLayout : public TensorViewLayout
        {
        protected:
            std::shared_ptr<Buffer> m_buffer;
            Strides                 m_strides;
            size_t                  m_offset;
        };
    }
}
