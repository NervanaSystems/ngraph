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

namespace ngraph
{
    namespace op
    {
        class Node;

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
        ///
        /// ## Implementation Status
        ///
        /// | Backend | Status             |
        /// | ------- | ------------------ |
        /// | NGVM    | Fully implemented. |
        class GetTupleElement : public Builtin
        {
        public:
            /// \brief Constructs a get-tuple-element operation.
            ///
            /// \param arg The input tuple.
            /// \param n The index of the tuple element to get.
            GetTupleElement(const std::shared_ptr<Node>& arg, size_t n)
                : Builtin({arg})
                , m_n{n}
            {
            }

            virtual void propagate_types() override;
            virtual std::string description() const override { return "GetTupleElement"; }
            /// \return The index of the tuple element to get.
            size_t get_n() const { return m_n; }
        protected:
            size_t m_n;
        };
    }
}
