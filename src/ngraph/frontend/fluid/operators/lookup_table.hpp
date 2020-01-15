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
        /// \brief Fluid lookup_table2
        class NGRAPH_API LookupTable2 : public ngraph::op::util::FusedOp
        {
        public:
            static constexpr NodeTypeInfo type_info{"FluidLookupTable2", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            LookupTable2() = default;
            /// \brief Constructs a LookupTable2 operation.
            ///
            /// \param w Input weight table
            /// \param ids  look up ids
            LookupTable2(const Output<Node>& w, const Output<Node>& ids, const int64_t padding_idx);

            virtual NodeVector decompose_op() const override;

            virtual void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            int64_t get_padding_idx() const { return m_padding_idx; }
        protected:
            int64_t m_padding_idx{-1};
        };

        /// \brief Fluid reduce_sum_grad
        class NGRAPH_API LookupTable2Grad : public ngraph::op::util::FusedOp
        {
        public:
            static constexpr NodeTypeInfo type_info{"FluidLookupTable2Grad", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            LookupTable2Grad() = default;

            /// \brief Constructs a LookupTable2Grad operation.
            ///
            /// \param w Input weight table
            /// \param ids Input lookup ids
            /// \param dout Input delta
            LookupTable2Grad(const Output<Node>& w,
                             const Output<Node>& ids,
                             const Output<Node>& dout);

            virtual NodeVector decompose_op() const override;

            virtual void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
