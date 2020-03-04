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

#include "tile.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Tile::type_info;

op::Tile::Tile(const Output<Node>& data, const Output<Node>& repeats)
    : Op({data, repeats})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Tile::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::Tile::validate_and_infer_types()
{
    auto arg_et = get_input_element_type(0);

    // Repeats should have integer data type. For now we only allow i64
    auto repeats_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          repeats_et.compatible(element::Type_t::i64),
                          "Tile repeats must have element type i64, but has ",
                          repeats_et);

    auto arg_shape = get_input_partial_shape(0);
    auto repeats_shape = get_input_partial_shape(1);
    auto repeats_rank = repeats_shape.rank();

    NODE_VALIDATION_CHECK(this, repeats_rank.compatible(1), "Shape of repeats must be of rank 1");

    auto out_shape = PartialShape::dynamic();

    if (auto const_repeats = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        if (arg_shape.is_static())
        {
            auto data_shape = arg_shape.to_shape();
            auto data_rank = data_shape.size();
            auto repeats_val = const_repeats->get_vector<int64_t>();
            auto repeats_rank = repeats_val.size();
            auto output_rank = std::max(data_rank, repeats_rank);

            // expand data shape and repeats to output rank
            data_shape.insert(data_shape.begin(), output_rank - data_rank, 1);
            repeats_val.insert(repeats_val.begin(), output_rank - repeats_rank, 1);

            Shape output_shape(output_rank);
            for (size_t i = 0; i < output_rank; i++)
            {
                output_shape[i] = data_shape[i] * repeats_val[i];
            }
            set_output_type(0, arg_et, output_shape);
        }
        else
        {
            set_output_type(0, arg_et, out_shape);
        }
    }
    else
    {
        set_output_type(0, arg_et, out_shape);
    }

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
}

shared_ptr<Node> op::Tile::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Tile>(new_args.at(0), new_args.at(1));
}

// TODO: This function is not implemented!
void op::Tile::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                 const OutputVector& /* deltas */)
{
    throw ngraph_error("generate_adjoints not implemented for Tile");
}
