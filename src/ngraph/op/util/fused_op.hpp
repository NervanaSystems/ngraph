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
        namespace util
        {
            /// \brief Abstract base class for fused ops, i.e ops that can be broken down into core ngraph ops
            ///
            class FusedOp : public Op
            {
            public:
                /// \brief Decomposes the FusedOp into a sub-graph consisting of core ngraph ops
                ///
                /// \return A vector of nodes comprising the sub-graph. The order of output
                ///         tensors must match the match output tensors of the FusedOp
                virtual NodeVector decompose_op() const = 0;

                void validate_and_infer_types() override;

                /// Pre and post validation hooks for op-specific actions
                virtual void pre_validate_and_infer_types() {}
                virtual void post_validate_and_infer_types() {}
                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const NodeVector& deltas) override;

            protected:
                FusedOp();

                /// \brief Constructs a FusedOp
                ///
                /// \param args Nodes that produce the input tensors for the fused op
                FusedOp(const NodeVector& args);

                FusedOp(const OutputVector& args);

                /// \brief Constructs a FusedOp
                ///
                /// \param args Nodes that produce the input tensors for the fused op
                FusedOp(const std::string& node_type, const NodeVector& args);
            };
        }
    }
}
