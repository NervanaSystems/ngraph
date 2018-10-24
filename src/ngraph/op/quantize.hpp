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
        /// \brief Quantize operation
        ///        Maps real input (r) to quantized output (q) using scale (s), offset (o) and round mode:
        ///        q = ROUND(r / s) + o
        class Quantize : public ngraph::op::Op
        {
        public:
            enum class RoundMode
            {
                // round to nearest integer
                // in case of two equidistant integers round away from zero e.g.
                // 2.5 -> 3
                // -3.5 -> -4
                ROUND_NEAREST_TOWARD_INFINITY,
                HALF_AWAY_FROM_ZERO, // TF mode for backward compatability

                // round to nearest integer
                // in case of two equidistant integers round toward zero e.g.
                // 2.5 -> 2
                // -3.5 -> -3
                ROUND_NEAREST_TOWARD_ZERO,

                // round to nearest integer
                // in case of two equidistant integers round up e.g.
                // 2.5 -> 3
                // -3.5 -> -3
                ROUND_NEAREST_UPWARD,

                // round to nearest integer
                // in case of two equidistant integers round down e.g.
                // 2.5 -> 2
                // -3.5 -> -4
                ROUND_NEAREST_DOWNWARD,

                // round to nearest integer
                // in case of two equidistant integers round to even e.g.
                // 2.5 -> 2
                // -3.5 -> -4
                ROUND_NEAREST_TOWARD_EVEN,

                // round to nearest integer away from zero
                ROUND_TOWARD_INFINITY,

                // round to nearest integer toward zero
                ROUND_TOWARD_ZERO,

                // round to nearest integer toward infinity (ceiling)
                ROUND_UP,

                // round to nearest integer toward negative infinity (floor)
                ROUND_DOWN,
            };

            /// \brief Constructs a Quantize operation
            /// \param input real input
            /// \param scale scale used for mapping
            /// \param offset offset used for mapping
            /// \param type output element type
            /// \param axes axis positions on which `scale` and `offset` are specified
            /// \param round_mode describes how to perform ROUND function (see above)
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
            RoundMode get_round_mode() const { return m_round_mode; }
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
