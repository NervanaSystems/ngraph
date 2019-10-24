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

namespace ngraph
{
    namespace op
    {
        /// \brief Elementwise type conversion operation.
        class Convert : public Op
        {
        public:
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"Convert", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            /// \brief Constructs a conversion operation.
            Convert() = default;
            /// \brief Constructs a conversion operation.
            ///
            /// \param arg          Node that produces the input tensor.
            /// \param element_type Element type for the output tensor.
            Convert(const Output<Node>& arg, const ngraph::element::Type& element_type);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const element::Type& get_convert_element_type() const { return m_element_type; }
            void set_convert_element_type(const element::Type& element_type)
            {
                m_element_type = element_type;
            }

        protected:
            ngraph::element::Type m_element_type;
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
        };
    }
}
