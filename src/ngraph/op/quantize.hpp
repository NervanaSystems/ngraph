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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Quantize operation
        ///        Maps real input (r) to quantized output (q) using scale (s), zero point (z) and
        ///        round mode: q = ROUND(r / s) + o
        class Quantize : public ngraph::op::Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            using RoundMode = op::RoundMode;

            /// \brief Constructs a Quantize operation
            /// \param input real input
            /// \param scale scale used for mapping
            /// \param zero_point zero point used for mapping
            /// \param type output element type
            /// \param axes axis positions on which `scale` and `zero_point` are specified
            /// \param round_mode describes how to perform ROUND function (see above)
            Quantize(const Output<Node>& input,
                     const Output<Node>& scale,
                     const Output<Node>& zero_point,
                     const ngraph::element::Type& type,
                     const ngraph::AxisSet& axes,
                     RoundMode round_mode);

            Quantize() = default;

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
