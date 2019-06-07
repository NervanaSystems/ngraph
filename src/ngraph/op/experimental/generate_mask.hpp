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
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a GenerateMask node with a given shape, seed,
            /// probability and training/inference mode
            GenerateMask();
            /// \brief Constructs a GenerateMask node with a given shape, seed,
            /// probability and training/inference mode
            GenerateMask(const std::shared_ptr<Node>& training,
                         const Shape& shape,
                         const element::Type& element_type,
                         uint64_t seed,
                         double prob,
                         bool use_seed = false);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \brief Returns the probability of a trial generating 1 (i.e. an element being kept)
            double get_probability() const { return m_probability; }
            void set_probability(double probability) { m_probability = probability; }
            /// \brief Returns the seed value supplied to a random generator
            uint64_t get_seed() const { return m_seed; }
            void set_seed(uint64_t seed) { m_seed = seed; }
            bool get_use_seed() const { return m_use_seed; }
            void set_use_seed(bool use_seed) { m_use_seed = use_seed; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override
            {
            }

            void validate_and_infer_types() override;
            Shape m_shape;
            element::Type m_element_type;
            bool m_use_seed{false};
            uint64_t m_seed{0};
            double m_probability{0.0};
        };
    }
}
