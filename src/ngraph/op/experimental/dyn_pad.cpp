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

#include "ngraph/op/experimental/dyn_pad.hpp"

using namespace std;
using namespace ngraph;

op::DynPad::DynPad(const std::shared_ptr<Node>& arg,
            const std::shared_ptr<Node>& padding_below,
            const std::shared_ptr<Node>& padding_above,
            const std::shared_ptr<Node>& padding_value)
    : Op("DynPad", check_single_output_args({arg, padding_below, padding_above, padding_value}))
{
    constructor_validate_and_infer_types();
}

void op::DynPad::validate_and_infer_types()
{
    // shape node should have integer data type. For now we only allow i64
    //TODO: potenially make the type more flexible to include other integer types
    auto padding_below_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          padding_below_et.compatible(element::Type_t::i64),
                          "DynPad shape must have element type i64, but has ",
                          padding_below_et);

    auto padding_above_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          padding_above_et.compatible(element::Type_t::i64),
                          "DynPad shape must have element type i64, but has ",
                          padding_above_et);

    // - padding_value is of scalar shape or rank is unknown
    auto padding_value_rank = get_input_partial_shape(2).rank();
    NODE_VALIDATION_CHECK(this,
                          padding_value_rank.is_dynamic() || padding_value_rank.compatible(0),
                          "DynPad arg is not scalar (rank = 0), but has rank = ",
                          padding_value_rank);
    

    
    auto arg_shape = get_input_partial_shape(0);
    auto arg_rank = arg_shape.rank();
    auto pd_bl_shape = get_input_partial_shape(1);
    auto pd_bl_rank = pd_bl_shape.rank();
    auto pd_ab_shape = get_input_partial_shape(2);
    auto pd_ab_rank = pd_bl_shape.rank();

    auto out_shape = PartialShape::dynamic();

    // Shapes of padding_below/above and arg must be rank compatible
    // Merge all ranks into the output shape. Also checks if ranks are compatible. 
    NODE_VALIDATION_CHECK(this,
                          out_shape.merge_rank(arg_rank) && 
                          out_shape.merge_rank(pd_bl_rank) && 
                          out_shape.merge_rank(pd_ab_rank),
                          "DynPad tensor and padding shapes are not of compatible ranks. arg, pd_bl, pd_ab = ",
                          arg_rank, pd_bl_rank, pd_ab_rank);

    // Infer output shape from input shapes.
    // We only infer forward here. 
    // TODO: It is possible to infer side-ways for other args. E.g. if arg_shape is fully static, then we know the rank of padding shapes .. etc. 
    out_shape = arg_shape + pd_bl_shape;
    out_shape = out_shape + pd_ab_shape;

    set_output_type(0, get_input_element_type(0), out_shape);
}

shared_ptr<Node> op::DynPad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynPad>(new_args.at(0), new_args.at(1), new_args.at(2));
}

// TODO: This function is not implemented!
void op::DynPad::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("generate_adjoints not implemented for DynPad");
}