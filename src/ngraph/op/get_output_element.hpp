/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Operation to get an element from a tuple.
        ///
        /// ## Parameters
        ///
        /// |     | Description                                                        |
        /// | --- | ------------------------------------------------------------------ |
        /// | `n` | The position of the element (0-based) to get from the input tuple. |
        ///
        /// ## Inputs
        ///
        /// |        | Type                                                        | Description                                |
        /// | ------ | ----------------------------------------------------------- | ------------------------------------------ |
        /// | `arg`  | \f$(T_1,\dots,T_{n-1},T_n,T_{n+1},\dots,T_m)~(m \geq 1)\f$ | An input tuple with at least `n` elements. |
        ///
        /// ## Output
        ///
        /// | Type      | Description                           |
        /// | --------- | ------------------------------------- |
        /// | \f$T_n\f$ | The `n`th element of the input tuple. |
        class GetOutputElement : public Node
        {
        public:
            /// \brief Constructs a get-tuple-element operation.
            ///
            /// \param arg The input tuple.
            /// \param n The index of the tuple element to get.
            GetOutputElement(const std::shared_ptr<Node>& arg, size_t n);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The index of the tuple element to get.
            size_t get_n() const { return m_n; }
            virtual NodeVector get_arguments() override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            size_t m_n;
        };
    }
}
