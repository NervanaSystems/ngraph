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
        /// \brief Operation which "adds" axes to an input tensor, replicating elements from the input as needed along the new axes.
        ///
        /// Informally, a broadcast "adds" axes to the input tensor, replicating elements from the input tensor as needed to fill the new dimensions.
        /// The parameter `m_broadcast_axes` indicates which of the output axes is being so added. For example, an output shape of `{2,5,6,2,8}` and
        /// broadcast axes of `{1,3,4}` means that the input must have shape `{2,6}`.
        ///
        /// Formally, given a shape or coordinate \f$S = [d_1,\dots,d_n]\f$ and a set of axis indices \f$A\f$, define \f$\textit{del}(S,A)\f$ to be
        /// the shape or coordinate obtained by deleting the \f$(a + 1)\f$th dimension from \f$S\f$ for each \f$a \in A\f$. Then given an input
        /// tensor \f$T\f$ of shape \f$\textit{del}(S,A)\f$ with element type \f$E\f$, broadcasting axes \f$A\f$ produces a tensor \f$T'\f$ of shape
        /// \f$S\f$ with element type \f$E\f$, where \f$T'[i_1,\dots,i_n] = T[del([i_1,\dots,i_n],A)]\f$.
        ///
        /// ## Parameters
        ///
        /// |                  | Description                                                              |
        /// | ---------------- | ------------------------------------------------------------------------ |
        /// | `shape`          | The shape \f$[d_1,\dots,d_n]\f$ of the broadcasted output.               |
        /// | `broadcast_axes` | The indices \f$A\f$ in the `shape` of each broadcasted (i.e., new) axis. |
        ///
        /// ## Inputs
        ///
        /// |       | Type                                                | Description                             |
        /// | ----- | --------------------------------------------------- | --------------------------------------- |
        /// | `arg` | \f$E[\mathit{del}([d_1,\dots,d_n],A)]~(n \geq 0)\f$ | A tensor of any shape and element type. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                     |
        /// | ---------------------- | ------------------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | The tensor \f$T'\f$, where \f$T'[i_1,\dots,i_n] = T[del([i_1,\dots,i_n],A)]\f$. |
        class Broadcast : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a conversion operation.
            ///
            /// \param arg            Node that produces the input tensor to be broadcast.
            /// \param shape          The shape of the output tensor.
            /// \param broadcast_axes The axis positions (0-based) in the result that are being broadcast. The
            ///                        remaining axes in shape must be the same as the shape of arg.
            Broadcast(const std::shared_ptr<Node>& arg,
                      const Shape& shape,
                      const AxisSet& broadcast_axes);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 1)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<Broadcast>(new_args.at(0), m_shape, m_broadcast_axes);
            }

            /// \return A set containing the indices of the broadcast axes (0-based).
            const AxisSet& get_broadcast_axes() const { return m_broadcast_axes; }
            const Shape& get_broadcast_shape() const { return m_shape; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;

            Shape m_shape;
            AxisSet m_broadcast_axes;
        };
    }
}
