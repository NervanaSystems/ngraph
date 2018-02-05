// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include "ngraph/ops/op.hpp"
#include "ngraph/types/type.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Elementwise type conversion operation.
        ///
        /// Each scalar in the input tensor is converted to the specified output element type. Note that the conversion may
        /// result in loss of precision. For example, conversion from `float32` to `int32` is allowed.
        ///
        /// ## Parameters
        ///
        /// |                | Description                              |
        /// | -------------- | ---------------------------------------- |
        /// | `element_type` | The element type \f$E'\f$ to convert to. |
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                               |
        /// | ----- | --------------------------------- | ----------------------------------------- |
        /// | `arg` | \f$E[d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and element type.   |
        ///
        /// ## Output
        ///
        /// | Type                    | Description                                                                                               |
        /// | ----------------------- | --------------------------------------------------------------------------------------------------------- |
        /// | \f$E'[d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n] = \mathit{convert}_{(E,E')}(\texttt{arg}[i_1,\dots,i_n])\f$ |
        class Convert : public UnaryElementwise
        {
        public:
            /// \brief Constructs a conversion operation.
            ///
            /// \param arg          Node that produces the input tensor.
            /// \param element_type Element type for the output tensor.
            Convert(const std::shared_ptr<Node>& arg, const ngraph::element::Type& element_type);

            virtual std::shared_ptr<Node> copy_with_new_args(const Nodes& new_args) const override
            {
                if (new_args.size() != 1)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<Convert>(new_args.at(0), m_element_type);
            }

            const element::Type& get_convert_element_type() const { return m_element_type; }
        protected:
            const ngraph::element::Type m_element_type;
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;
        };
    }
}
