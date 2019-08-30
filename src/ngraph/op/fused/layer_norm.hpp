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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Layer Normalization
        ///
        class LayerNorm : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            LayerNorm() = default;
            /// \brief Constructs an LayerNorm operation.
            ///
            /// \param data Input tensor
            /// \param scale Scale tensor
            /// \param bias Bias tensor
            /// \param epsilon Small number to add for stability of rsqrt
            /// \param begin_norm_axis Axis where normalization starts, default - -1
            LayerNorm(const Output<Node>& data,
                      const Output<Node>& scale,
                      const Output<Node>& bias,
                      double epsilon,
                      int64_t begin_norm_axis = -1);

            LayerNorm(const Output<Node>& data,
                      double epsilon,
                      int64_t begin_norm_axis = -1);

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
        private:
            double m_epsilon;
            int64_t m_begin_norm_axis{-1};
        };
        /// \brief Layer Normalization Backprop
        ///
        class LayerNormBackprop : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            LayerNormBackprop() = default;
            /// \brief Constructs an LayerNormBackprop operation.
            LayerNormBackprop(const Output<Node>& data,
                              const Output<Node>& mean,
                              const Output<Node>& variance,
                              const Output<Node>& delta);
            LayerNormBackprop(const Output<Node>& data,
                              const Output<Node>& mean,
                              const Output<Node>& variance,
                              const Output<Node>& delta
                              const Output<Node>& scale
                              const Output<Node>& bias);
        };
    }
}
