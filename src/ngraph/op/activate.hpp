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
        /// \brief Activate
        ///
        class Activate : public op::Op
        {
        public:
            Activate(const std::shared_ptr<State>& state)
                : Op("Activate", {})
                , m_state(state)
            {
                constructor_validate_and_infer_types();
            }

            Activate(const std::shared_ptr<Node> arg, const std::shared_ptr<State>& state)
                : Op("Activate", {check_single_output_args({arg})})
                , m_state(state)
            {
                constructor_validate_and_infer_types();
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            std::shared_ptr<State> get_state() const { return m_state; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override
            {
            }

            void virtual validate_and_infer_types() override
            {
                set_output_type(0, element::i8, Shape{});
            }
            std::shared_ptr<State> m_state;
        };
    }
}
