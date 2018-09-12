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
            GenerateMask(const std::shared_ptr<Node>& training,
                         const std::shared_ptr<Node>& activate,
                         const Shape& shape,
                         const element::Type& element_type,
                         double prob,
                         const std::shared_ptr<RNGState>& state)
                : Op("GenerateMask", {check_single_output_args({training, activate})})
                , m_state(state)
                , m_shape(shape)
                , m_element_type(element_type)
                , m_probability(prob)
            {
                constructor_validate_and_infer_types();
            }

            //Activate node is added as a control dependency
            GenerateMask(std::shared_ptr<Node> training,
                         const Shape& shape,
                         const element::Type& element_type,
                         double prob,
                         const std::shared_ptr<RNGState>& state)
                : Op("GenerateMask", {check_single_output_args({training})})
                , m_state(state)
                , m_shape(shape)
                , m_element_type(element_type)
                , m_probability(prob)
            {
                constructor_validate_and_infer_types();
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            std::shared_ptr<RNGState> get_state() const { return m_state; }
            std::shared_ptr<Node> get_training_node() const { return get_argument(0); }
            double get_probability() const { return m_probability; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override
            {
            }

            void validate_and_infer_types() override;
            std::shared_ptr<RNGState> m_state;
            Shape m_shape;
            element::Type m_element_type;
            double m_probability = 1.;
        };
    }
}
