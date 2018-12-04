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
        /// \brief Dequantize operation
        ///        Maps quantized input (q) to real output (r) using scale (s) and offset (o):
        ///        r = (q - o) * s
        class Dequantize : public ngraph::op::Op
        {
        public:
            /// \brief Constructs a Dequantize operation
            /// \param input quantized input
            /// \param scale scale used for mapping
            /// \param offset offset used for mapping
            /// \param type output element type
            /// \param axes axis positions on which `scale` and `offset` are specified
            Dequantize(std::shared_ptr<Node> input,
                       std::shared_ptr<Node> scale,
                       std::shared_ptr<Node> offset,
                       const ngraph::element::Type& type,
                       const ngraph::AxisSet& axes);

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
        };
    }
}
