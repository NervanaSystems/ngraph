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

#include "ngraph/frontend/fluid/operators/lookup_table.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph::fluid;

constexpr NodeTypeInfo LookupTable2::type_info;

LookupTable2::LookupTable2(const Output<Node>& w,
                           const Output<Node>& ids,
                           const int64_t padding_idx)
    : FusedOp({w, ids})
    , m_padding_idx(padding_idx)
{
    constructor_validate_and_infer_types();
}

NodeVector LookupTable2::decompose_op() const
{
    auto w = input_value(0);
    auto ids = input_value(1);
    auto padding_idx = get_padding_idx();

    auto table_shape = get_input_shape(0);

    NODE_VALIDATION_CHECK(
        this, table_shape.size() == 2, "The dimension of look up table must be 2");

    auto row_number = table_shape[0];

    auto masked_w = w;

    if (padding_idx != -1)
    {
        vector<size_t> mask(row_number, 1);
        mask[padding_idx] = 0;

        auto mask_node = make_shared<op::Constant>(w.get_element_type(), Shape{row_number}, mask);
        auto mask_bcast = make_shared<op::Broadcast>(mask_node, table_shape, AxisSet{1});
        masked_w = w * mask_bcast;
    }

    auto out = make_shared<op::Gather>(masked_w, ids);

    return {out};
}

shared_ptr<Node> LookupTable2::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);

    return make_shared<LookupTable2>(new_args.at(0), new_args.at(1), get_padding_idx());
}

void LookupTable2::pre_validate_and_infer_types()
{
    auto pshape_w = get_input_partial_shape(0);
    auto pshape_ids = get_input_partial_shape(1);

    if (pshape_w.is_dynamic() || pshape_ids.is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

constexpr NodeTypeInfo LookupTable2Grad::type_info;

LookupTable2Grad::LookupTable2Grad(const Output<Node>& w,
                                   const Output<Node>& ids,
                                   const Output<Node>& dout)
    : FusedOp({w, ids, dout})
{
    constructor_validate_and_infer_types();
}

void LookupTable2Grad::pre_validate_and_infer_types()
{
    if (get_input_partial_shape(0).is_dynamic() || get_input_partial_shape(1).is_dynamic() ||
        get_input_partial_shape(2).is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> LookupTable2Grad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<LookupTable2Grad>(new_args.at(0), new_args.at(1), new_args.at(2));
}

NodeVector LookupTable2Grad::decompose_op() const
{
    return {};
}
