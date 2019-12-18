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

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace fluid
    {
        /// \brief Fluid layout converter
        class NGRAPH_API LayoutConverter : public ngraph::op::util::FusedOp
        {
        public:
            static constexpr NodeTypeInfo type_info{"FluidLayoutConverter", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            LayoutConverter() = default;
            /// \brief Constructs a LayoutConverter operation.
            ///
            /// \param x Input x
            /// \param mode : 1. nhwc->nchw, 2 hchw->nhwc
            LayoutConverter(const Output<Node>& x, const int mode);

            virtual NodeVector decompose_op() const override;

            virtual void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            int get_mode() const { return m_mode; }
        protected:
            int m_mode;
        };
    } // namespace fluid
} // namespace ngraph
