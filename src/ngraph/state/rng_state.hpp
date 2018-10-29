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

#include <functional>
#include <memory>
#include <random>

#include "state.hpp"

namespace ngraph
{
    //can be based on TensorSate to cache values instead of just caching seed
    class RNGState : public State
    {
    public:
        static RNGState* create_rng_state(unsigned int seed, double probability)
        {
            auto rng = new RNGState(seed, probability);
            return rng;
        }

        RNGState(unsigned int seed, double probability)
            : State()
            , m_generator(seed)
            , m_distribution(probability)
        {
        }
        virtual void activate() override;
        virtual void deactivate() override;
        virtual ~RNGState() {}
        std::mt19937& get_generator() { return m_generator; }
        std::bernoulli_distribution& get_distribution() { return m_distribution; }
    protected:
        std::mt19937 m_generator;
        std::bernoulli_distribution m_distribution;
    };
}
