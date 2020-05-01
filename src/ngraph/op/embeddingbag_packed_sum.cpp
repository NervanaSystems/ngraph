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

#include "ngraph/op/embeddingbag_packed_sum.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::EmbeddingBagPackedSum::type_info;

op::v3::EmbeddingBagPackedSum::EmbeddingBagPackedSum(const Output<Node>& emb_table,
                                                     const Output<Node>& indices,
                                                     const Output<Node>& per_sample_weights)
    : Op({emb_table, indices, per_sample_weights})
{
    constructor_validate_and_infer_types();
}

op::v3::EmbeddingBagPackedSum::EmbeddingBagPackedSum(const Output<Node>& emb_table,
                                                     const Output<Node>& indices)
    : Op({emb_table, indices})
{
    constructor_validate_and_infer_types();
}

void op::v3::EmbeddingBagPackedSum::validate_and_infer_types()
{
    enum
    {
        EMB_TABLE,
        INDICES,
        PER_SAMPLE_WEIGHTS,
    };

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(INDICES) == element::i64 ||
                              get_input_element_type(INDICES) == element::i32,
                          "INDICES type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(INDICES).is_dynamic() ||
                              get_input_partial_shape(INDICES).to_shape().size() == 2,
                          "INDICES must be 2D");

    if (get_input_size() == 3)
    {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(EMB_TABLE).compatible(
                                  get_input_element_type(PER_SAMPLE_WEIGHTS)),
                              "Per sample weight element type (",
                              get_input_element_type(PER_SAMPLE_WEIGHTS),
                              ") must match embedding table element type (",
                              get_input_element_type(EMB_TABLE),
                              ")");

        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(PER_SAMPLE_WEIGHTS).is_dynamic() ||
                                  get_input_partial_shape(PER_SAMPLE_WEIGHTS).to_shape().size() ==
                                      2,
                              "PER_SAMPLE_WEIGHTS must be 2D");

        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(INDICES).compatible(
                                  get_input_partial_shape(PER_SAMPLE_WEIGHTS)),
                              "INDICES and PER_SAMPLE_WEIGHTS shape must be same");
    }

    element::Type result_et = get_input_element_type(EMB_TABLE);

    const PartialShape& emb_table_shape = get_input_partial_shape(EMB_TABLE);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    PartialShape result_shape;
    if (emb_table_shape.rank().is_static())
    {
        std::vector<Dimension> result_dims(emb_table_shape.rank().get_length());
        result_dims[0] = indices_shape.rank().is_static() ? indices_shape[0] : Dimension::dynamic();
        for (size_t i = 1; i < emb_table_shape.rank().get_length(); i++)
        {
            result_dims[i] = emb_table_shape[i];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node>
    op::v3::EmbeddingBagPackedSum::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() == 2)
    {
        return make_shared<EmbeddingBagPackedSum>(new_args.at(0), new_args.at(1));
    }
    else
    {
        return make_shared<EmbeddingBagPackedSum>(new_args.at(0), new_args.at(1), new_args.at(2));
    }
}
