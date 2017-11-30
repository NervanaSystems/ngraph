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

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Operation to construct a tuple.
        ///
        /// ## Inputs
        ///
        /// |        | Type                           | Description                            |
        /// | ------ | ------------------------------ | -------------------------------------- |
        /// | `args` | \f$T_1,\dots,T_n~(n \geq 0)\f$ | The elements of the constructed tuple. |
        ///
        /// ## Output
        ///
        /// | Type                  | Description                                                |
        /// | --------------------- | ---------------------------------------------------------- |
        /// | \f$(T_1,\dots,T_n)\f$ | The tuple \f$(\texttt{args}[0],\dots,\texttt{args}[n])\f$. |
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status             |
        /// | ------- | ------------------ |
        /// | NGVM    | Fully implemented. |
        class Tuple : public Node
        {
        public:
            /// \brief Constructs a tuple construction operation.
            ///
            /// \param args The nodes that produce the elements of the constructed tuple.
            Tuple(const Nodes& args);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                return std::make_shared<Tuple>(new_args);
            }
        };
    }
}
