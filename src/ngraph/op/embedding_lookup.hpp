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

#include "ngraph/axis_set.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/index_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Returns embeddings for given indices
        class EmbeddingLookup : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a EmbeddingLookup operation.
            EmbeddingLookup() = default;
            /// \brief Constructs a EmbeddingLookup operation.
            ///
            /// EmbeddingLookup constructs an output tensor by replacing every index in a given
            /// input tensor with a row (from the weights matrix) at that index
            ///
            /// \param data The input indices for tokens to be translated into embeddings
            /// \param weights is a dense matrix [N,M] where each row 0..N
            /// corresponds to an embedding (i.e. typically, a vector of real numbers) of length M
            EmbeddingLookup(const Output<Node>& data, const Output<Node>& weights)
                : Op({data, weights})
            {
                constructor_validate_and_infer_types();
            }

            void validate_and_infer_types() override;

            void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) override
            {
                throw ngraph_error("Not yet implemented");
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
