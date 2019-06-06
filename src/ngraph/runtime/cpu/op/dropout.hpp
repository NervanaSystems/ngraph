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

#include "ngraph/op/op.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class Dropout : public Op
        {
        public:
            Dropout(const std::shared_ptr<Node>& input,
                    const std::shared_ptr<Node>& gm_const,
                    const uint32_t seed,
                    const double value); // value = 1 - dropout_prob

            uint32_t get_seed() const { return m_seed; }
            double get_value() const { return m_value; } // this value is 1- probability
            void set_seed(uint32_t new_seed) { m_seed = new_seed; }
            void set_value(double new_value) { m_value = new_value; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            //virtual void generate_adjoints(autodiff::Adjoints& adjoints,
            //                               const NodeVector& deltas) override;

        private:
            uint32_t m_seed;
            double m_value;
        };
    }
}
