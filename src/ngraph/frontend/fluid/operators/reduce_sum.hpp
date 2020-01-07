//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace fluid
    {
        /// \brief Fluid reduce_sum
        class NGRAPH_API ReduceSum : public ngraph::op::util::FusedOp
        {
        public:
            static constexpr NodeTypeInfo type_info{"FluidReduceSum", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            ReduceSum() = default;
            /// \brief Constructs a ReduceSum operation.
            ///
            /// \param data Input tensor
            ReduceSum(const Output<Node>& data,
                      const vector<int>& dim,
                      bool reduce_all,
                      bool keep_dim);

            virtual NodeVector decompose_op() const override;

            virtual void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            vector<int> m_dim;
            AxisSet m_reduction_axes;
            bool m_reduce_all;
            bool m_keep_dim;
        };

        /// \brief Fluid reduce_sum_grad
        class NGRAPH_API ReduceSumGrad : public ngraph::op::util::FusedOp
        {
        public:
            static constexpr NodeTypeInfo type_info{"FluidReduceSumGrad", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            ReduceSumGrad() = default;

            /// \brief Constructs a ReduceSumGrad operation.
            ///
            /// \param data Input tensor
            ReduceSumGrad(const Output<Node>& x,
                          const Output<Node>& y,
                          const vector<int>& dim,
                          bool reduce_all,
                          bool keep_dim);

            virtual NodeVector decompose_op() const override;

            virtual void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            vector<int> m_dim;
            AxisSet m_reduction_axes;
            bool m_reduce_all;
            bool m_keep_dim;
        };
    }
}
