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

#include <memory>
#include "ngraph/op/constant.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/state/rng_state.hpp"
namespace ngraph
{
    namespace op
    {
        /// \brief GenerateMask
        ///
        class GenerateMask : public op::Op
        {
        public:
            /// \brief Constructs a GenerateMask node with a given shape, sed,
            /// probability and training/inference mode
            GenerateMask(const std::shared_ptr<Node>& training,
                         const Shape& shape,
                         const element::Type& element_type,
                         unsigned int seed,
                         double prob);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \brief Returns the probability of a trial generating 1 (i.e. an element being kept)
            double get_probability() const { return m_probability; }
            /// \brief Returns the seed value supplied to a random generator
            unsigned int get_seed() const { return m_seed; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override
            {
            }

            void validate_and_infer_types() override;
            Shape m_shape;
            element::Type m_element_type;
            unsigned int m_seed;
            double m_probability;
        };
    }
}
