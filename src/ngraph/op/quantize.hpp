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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace op
    {
        class Quantize : public ngraph::op::Op
        {
        public:
            enum class RoundMode
            {
                HALF_AWAY_FROM_ZERO,
                HALF_TO_EVEN
            };

            Quantize(std::shared_ptr<Node> input,
                     std::shared_ptr<Node> scale,
                     std::shared_ptr<Node> offset,
                     const ngraph::element::Type& type,
                     const ngraph::AxisSet& axes,
                     RoundMode round_mode);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const ngraph::AxisSet& get_axes() const { return m_axes; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        private:
            ngraph::element::Type m_type;
            ngraph::AxisSet m_axes;
            RoundMode m_round_mode;
        };
    }
}
