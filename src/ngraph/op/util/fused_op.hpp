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
            /// \brief Abstract base class for fused ops, i.e ops that can be broken down into core
            ///        ngraph ops
            ///
            class NGRAPH_API FusedOp : public Op
            {
            public:
                bool supports_decompose() const override { return true; }
                // Fused op decomposition can be performed in the presence of
                // partial shapes
                virtual bool can_decompose_with_partial_shapes() { return false; }
                // Shape inference that will use fused op decomposition to infer
                // shapes and types of output elements. Ops can choose to override
                // and provide a more direct implementation.
                void validate_and_infer_types() override;

                // Pre-validation hook that will be invoked before op
                // decomposition in validate_and_infer_types().
                // Can be used for attribute validation and setting types/shapes
                // that can be inferred without requiring op decomposition.
                // Can also be used to set shape specialization hints
                // (set_input_is_relevant_to_shape())
                virtual void pre_validate_and_infer_types() {}
                // Post-validation hook that will be invoked after op decomposition
                // in validate_and_infer_types().
                virtual void post_validate_and_infer_types() {}
                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;

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
