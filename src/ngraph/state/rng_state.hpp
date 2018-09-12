//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "state.hpp"

namespace ngraph
{
    //can be based on TensorSate to cache values instead of just caching seed
    class RNGState : public State
    {
    public:
        RNGState()
            : State()
        {
        }
        virtual void activate() override;
        virtual void deactivate() override;
        unsigned int get_seed() const { return m_seed; }
        virtual ~RNGState() {}
    protected:
        unsigned int m_seed = 0;
    };
}
