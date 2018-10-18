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
        ///        Maps real input (r) to quantized output (q) using scale (s), offset (o) and round mode
        ///        q = ROUND(r / s) - o
        class Quantize : public ngraph::op::Op
        {
        public:
            enum class RoundMode
            {
                // x.5 to x+1
                // -x.5 to -(x+1)
                // everything else to nearest integer
                //  2.25 ->  2.0
                //  2.50 ->  3.0
                //  2.75 ->  3.0
                // -2.25 -> -2.0
                // -2.50 -> -3.0
                // -2.75 -> -3.0
                //  3.25 ->  3.0
                //  3.50 ->  4.0
                //  3.75 ->  4.0
                // -3.25 -> -3.0
                // -3.50 -> -4.0
                // -3.75 -> -4.0
                HALF_AWAY_FROM_ZERO,

                // x.5 and -x.5 to nearest even integer
                // everything else to nearest integer
                //  2.25 ->  2.0
                //  2.50 ->  2.0
                //  2.75 ->  3.0
                // -2.25 -> -2.0
                // -2.50 -> -2.0
                // -2.75 -> -3.0
                //  3.25 ->  3.0
                //  3.50 ->  4.0
                //  3.75 ->  4.0
                // -3.25 -> -3.0
                // -3.50 -> -4.0
                // -3.75 -> -4.0
                HALF_TO_EVEN,

                // everything to next integer towards infinity
                //  2.25 ->  3.0
                //  2.50 ->  3.0
                //  2.75 ->  3.0
                // -2.25 -> -2.0
                // -2.50 -> -2.0
                // -2.75 -> -2.0
                //  3.25 ->  4.0
                //  3.50 ->  4.0
                //  3.75 ->  4.0
                // -3.25 -> -3.0
                // -3.50 -> -3.0
                // -3.75 -> -3.0
                ALL_TOWARD_POSITIVE_INFINITY,

                // everything to next integer towards -infinity
                //  2.25 ->  2.0
                //  2.50 ->  2.0
                //  2.75 ->  2.0
                // -2.25 -> -3.0
                // -2.50 -> -3.0
                // -2.75 -> -3.0
                //  3.25 ->  3.0
                //  3.50 ->  3.0
                //  3.75 ->  3.0
                // -3.25 -> -4.0
                // -3.50 -> -4.0
                // -3.75 -> -4.0
                ALL_TOWARD_NEGATIVE_INFINITY,

                // everything to next integer towards zero
                //  2.25 ->  2.0
                //  2.50 ->  2.0
                //  2.75 ->  2.0
                // -2.25 -> -2.0
                // -2.50 -> -2.0
                // -2.75 -> -2.0
                //  3.25 ->  3.0
                //  3.50 ->  3.0
                //  3.75 ->  3.0
                // -3.25 -> -3.0
                // -3.50 -> -3.0
                // -3.75 -> -3.0
                ALL_TOWARD_ZERO
            };

            /// \brief Constructs a Quantize operation
            /// \param input real input
            /// \param scale element type: same as `input`, shape: `input` shape projected along `axes`
            /// \param offset element type: same as `type`, shape: `input` shape projected along `axes`
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
