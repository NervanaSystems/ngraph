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

#include "ngraph/op/embedding_segments_sum.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::EmbeddingSegmentsSum::type_info;

op::v3::EmbeddingSegmentsSum::EmbeddingSegmentsSum(const Output<Node>& emb_table,
                                                   const Output<Node>& indices,
                                                   const Output<Node>& segment_ids,
                                                   const Output<Node>& num_segments,
                                                   const Output<Node>& per_sample_weights,
                                                   const Output<Node>& default_index)
    : Op({emb_table, indices, segment_ids, num_segments, per_sample_weights, default_index})
{
    constructor_validate_and_infer_types();
}

op::v3::EmbeddingSegmentsSum::EmbeddingSegmentsSum(const Output<Node>& emb_table,
                                                   const Output<Node>& indices,
                                                   const Output<Node>& segment_ids,
                                                   const Output<Node>& num_segments,
                                                   const Output<Node>& per_sample_weights)
    : Op({emb_table, indices, segment_ids, num_segments, per_sample_weights})
{
    constructor_validate_and_infer_types();
}

op::v3::EmbeddingSegmentsSum::EmbeddingSegmentsSum(const Output<Node>& emb_table,
                                                   const Output<Node>& indices,
                                                   const Output<Node>& segment_ids,
                                                   const Output<Node>& num_segments)
    : Op({emb_table, indices, segment_ids, num_segments})
{
    constructor_validate_and_infer_types();
}

void op::v3::EmbeddingSegmentsSum::validate_and_infer_types()
{
    enum
    {
        EMB_TABLE,
        INDICES,
        SEGMENT_IDS,
        NUM_SEGMENTS,
        PER_SAMPLE_WEIGHTS,
        DEFAULT_INDEX
    };

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(SEGMENT_IDS) == element::i64 ||
                              get_input_element_type(SEGMENT_IDS) == element::i32,
                          "SEGMENT_IDS type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(INDICES) == element::i64 ||
                              get_input_element_type(INDICES) == element::i32,
                          "INDICES type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(NUM_SEGMENTS) == element::i64 ||
                              get_input_element_type(NUM_SEGMENTS) == element::i32,
                          "NUM_SEGMENTS type must be i32 or i64");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(INDICES).compatible(get_input_element_type(SEGMENT_IDS)),
        "Segment_ids element type (",
        get_input_element_type(SEGMENT_IDS),
        ") must match indices element type (",
        get_input_element_type(INDICES),
        ")");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(SEGMENT_IDS).compatible(get_input_element_type(NUM_SEGMENTS)),
        "Num_segments element type (",
        get_input_element_type(NUM_SEGMENTS),
        ") must match Segment_ids element type (",
        get_input_element_type(SEGMENT_IDS),
        ")");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(INDICES).is_dynamic() ||
                              get_input_partial_shape(INDICES).to_shape().size() == 1,
                          "INDICES must be 1D");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(SEGMENT_IDS).is_dynamic() ||
                              get_input_partial_shape(SEGMENT_IDS).to_shape().size() == 1,
                          "SEGMENT_IDS must be 1D");

    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(INDICES).compatible(get_input_partial_shape(SEGMENT_IDS)),
        "INDICES and SEGMENT_IDS shape must be same");

    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(NUM_SEGMENTS).compatible(PartialShape{}),
                          "NUM_SEGMENTS must be a scalar");

    if (get_input_size() >= 5)
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
                                      1,
                              "PER_SAMPLE_WEIGHTS must be 1D");

        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(INDICES).compatible(
                                  get_input_partial_shape(PER_SAMPLE_WEIGHTS)),
                              "INDICES and PER_SAMPLE_WEIGHTS shape must be same");
    }

    if (get_input_size() == 6)
    {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(DEFAULT_INDEX) == element::i64 ||
                                  get_input_element_type(DEFAULT_INDEX) == element::i32,
                              "DEFAULT_INDEX type must be i32 or i64");

        NODE_VALIDATION_CHECK(
            this,
            get_input_element_type(INDICES).compatible(get_input_element_type(DEFAULT_INDEX)),
            "Default_index element type (",
            get_input_element_type(DEFAULT_INDEX),
            ") must match indices element type (",
            get_input_element_type(INDICES),
            ")");

        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(DEFAULT_INDEX).compatible(PartialShape{}),
                              "DEFAULT_INDEX must be a scalar");
    }

    element::Type result_et = get_input_element_type(EMB_TABLE);

    const PartialShape& emb_table_shape = get_input_partial_shape(EMB_TABLE);

    PartialShape result_shape;
    if (emb_table_shape.rank().is_static())
    {
        std::vector<Dimension> result_dims(emb_table_shape.rank().get_length());
        result_dims[0] = Dimension::dynamic();
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

    set_input_is_relevant_to_shape(NUM_SEGMENTS);
    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node>
    op::v3::EmbeddingSegmentsSum::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() == 4)
    {
        return make_shared<EmbeddingSegmentsSum>(
            new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
    }
    else if (new_args.size() == 5)
    {
        return make_shared<EmbeddingSegmentsSum>(
            new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
    }
    else
    {
        return make_shared<EmbeddingSegmentsSum>(new_args.at(0),
                                                 new_args.at(1),
                                                 new_args.at(2),
                                                 new_args.at(3),
                                                 new_args.at(4),
                                                 new_args.at(5));
    }
}
