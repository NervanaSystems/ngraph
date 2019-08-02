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

#include "tile.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

const string op::Tile::type_name{"Tile"};

op::Tile::Tile(const std::shared_ptr<Node>& arg, const std::shared_ptr<Node>& repeats)
    : Op(check_single_output_args({arg, repeats}))
{
    constructor_validate_and_infer_types();
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
    auto arg_rank = arg_shape.rank();
    auto repeats_shape = get_input_partial_shape(1);
    auto repeats_rank = repeats_shape.rank();

    auto output_rank = Rank::dynamic();
    NODE_VALIDATION_CHECK(this, repeats_rank.compatible(1), "Shape of repeats must be of rank 1");

    if (arg_rank.is_static())
    {
        // Repeats shapes should be of form {arg_rank} or dynamic
        NODE_VALIDATION_CHECK(this,
                              repeats_shape.compatible(PartialShape{arg_rank}),
                              "Arg and repeats ranks mismatch");

        output_rank = arg_rank;
    }

    auto out_shape = PartialShape::dynamic(output_rank);

    if (auto const_repeats = dynamic_pointer_cast<op::Constant>(get_argument(1)))
    {
        if (arg_shape.is_static())
        {
            auto shape = arg_shape.to_shape();
            auto repeats_val = const_repeats->get_vector<int64_t>();

            Shape output_shape(shape.size());
            for (size_t i = 0; i < shape.size(); i++)
            {
                output_shape[i] = shape[i] * repeats_val[i];
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

    set_input_is_relevant_to_shape(1);
}

shared_ptr<Node> op::Tile::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Tile>(new_args.at(0), new_args.at(1));
}

// TODO: This function is not implemented!
void op::Tile::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("generate_adjoints not implemented for Tile");
}
