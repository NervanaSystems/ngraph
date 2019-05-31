//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
        virtual ~RNGState() override {}
        std::mt19937& get_generator() { return m_generator; }
        std::bernoulli_distribution& get_distribution() { return m_distribution; }
    protected:
        std::mt19937 m_generator;
        std::bernoulli_distribution m_distribution;
    };

    // RNG with uniform real distribution
    class RNGUniformState : public State
    {
    public:
        static RNGUniformState* create_rng_state(unsigned int seed, double probability)
        {
            auto rng = new RNGUniformState(seed, probability);
            return rng;
        }

        RNGUniformState(unsigned int seed, double probability)
            : State()
            , m_seed(seed)
            , m_probability(probability) // don't really need this prob
            , m_distribution(0,1)
        {
            m_generator.seed(m_seed);
        }
        virtual void activate() override;
        virtual void deactivate() override;
        virtual ~RNGUniformState() override {}
        std::minstd_rand& get_generator() { return m_generator; }
        std::uniform_real_distribution<float>& get_distribution() { return m_distribution; }
        double get_probability() {return m_probability; } ;
    protected:
        double m_probability;
        unsigned int m_seed;
        std::minstd_rand m_generator;
        std::uniform_real_distribution<float> m_distribution;
    };
}
