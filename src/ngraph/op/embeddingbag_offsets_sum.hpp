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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/util/embeddingbag_offsets_base.hpp"
#include "ngraph/op/util/index_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            /// \brief Returns embeddings for given indices
            class NGRAPH_API EmbeddingBagOffsetsSum : public util::EmbeddingBagOffsetsBase
            {
            public:
                static constexpr NodeTypeInfo type_info{"EmbeddingBagOffsetsSum", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a EmbeddingBagOffsetsSum operation.
                EmbeddingBagOffsetsSum() = default;
                /// \brief Constructs a EmbeddingBagOffsetsSum operation.
                ///
                /// EmbeddingBagOffsetsSum constructs an output tensor by replacing every index in a
                /// given
                /// input tensor with a row (from the weights matrix) at that index
                ///
                /// \param emb_table tensor containing the embedding lookup table of the module of
                /// shape [num_emb, emb_dim1, emb_dim2, ...] and  of type T
                /// \param tensor of shape [num_indices] and of type T_IND. Required
                /// \param offsets tensor of shape [batch] and of type T_IND containing the starting
                /// index positions of each "bag" in indices. Required.
                /// \param default_index scalar of type T_IND containing default index in embedding
                /// table to fill empty "bags". If not provided empty "bags"
                /// are filled with zeros. Optional.
                /// \param per_sample_weigths tensor of the same shape as indices and of type T.
                /// Each value in this tensor are multiplied with each
                /// value pooled from embedding table for each index. Optional.

                EmbeddingBagOffsetsSum(const Output<Node>& emb_table,
                                       const Output<Node>& indices,
                                       const Output<Node>& offsets,
                                       const Output<Node>& default_index,
                                       const Output<Node>& per_sample_weights);

                EmbeddingBagOffsetsSum(const Output<Node>& emb_table,
                                       const Output<Node>& indices,
                                       const Output<Node>& offsets,
                                       const Output<Node>& default_index);

                EmbeddingBagOffsetsSum(const Output<Node>& emb_table,
                                       const Output<Node>& indices,
                                       const Output<Node>& offsets);

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
        using v3::EmbeddingBagOffsetsSum;
    }
}
