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
#include "ngraph/op/fused/gather_elements.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/gather_nd.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GatherElements::type_info;

op::GatherElements::GatherElements(const Output<Node>& arg1, const Output<Node>& arg2, int64_t axis)
    : FusedOp({arg1, arg2})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

NodeVector op::GatherElements::decompose_op() const
{
    auto axis = get_axis();
    auto data = input_value(0);
    auto indicate = input_value(1);

    auto x_shape = data.get_shape();
    auto indicate_shape = indicate.get_shape();

    std::shared_ptr<ngraph::Node> data_slice_1 =
        std::make_shared<op::Slice>(data, ngraph::Coordinate{0, 0}, ngraph::Coordinate{1, 2});

    std::shared_ptr<ngraph::Node> data_slice_2 =
        std::make_shared<op::Slice>(data, ngraph::Coordinate{0, 0}, ngraph::Coordinate{1, 2});

    auto ds1_shape = data_slice_1->get_shape();
    auto ds2_shape = data_slice_2->get_shape();

    std::shared_ptr<ngraph::Node> ind_slice_1 =
        std::make_shared<op::Slice>(indicate, ngraph::Coordinate{0, 0}, ngraph::Coordinate{1, 1});

    std::shared_ptr<ngraph::Node> ind_slice_2 =
        std::make_shared<op::Slice>(indicate, ngraph::Coordinate{0, 0}, ngraph::Coordinate{1, 1});

    std::shared_ptr<ngraph::Node> ind_slice_3 =
        std::make_shared<op::Slice>(indicate, ngraph::Coordinate{0, 0}, ngraph::Coordinate{1, 1});

    std::shared_ptr<ngraph::Node> ind_slice_4 =
        std::make_shared<op::Slice>(indicate, ngraph::Coordinate{0, 0}, ngraph::Coordinate{1, 1});

    auto convert1 = std::make_shared<ngraph::op::Convert>(ind_slice_1, ngraph::element::i64);
    auto convert2 = std::make_shared<ngraph::op::Convert>(ind_slice_2, ngraph::element::i64);
    auto convert3 = std::make_shared<ngraph::op::Convert>(ind_slice_3, ngraph::element::i64);
    auto convert4 = std::make_shared<ngraph::op::Convert>(ind_slice_4, ngraph::element::i64);

    auto indicate_shape_hard =
        op::Constant::create(data.get_element_type(), ngraph::Shape{1, 2}, {0});
    auto convert_indicate_shape_hard =
        std::make_shared<ngraph::op::Convert>(indicate_shape_hard, ngraph::element::i64);

    std::shared_ptr<ngraph::Node> gather_nd_1_slice_res =
        std::make_shared<op::GatherND>(data_slice_1, convert_indicate_shape_hard);
    std::shared_ptr<ngraph::Node> gather_nd_2_slice_res =
        std::make_shared<op::GatherND>(data_slice_1, convert_indicate_shape_hard);
    std::shared_ptr<ngraph::Node> gather_nd_3_slice_res =
        std::make_shared<op::GatherND>(data_slice_2, convert_indicate_shape_hard);
    std::shared_ptr<ngraph::Node> gather_nd_4_slice_res =
        std::make_shared<op::GatherND>(data_slice_2, convert_indicate_shape_hard);

    std::vector<std::shared_ptr<ngraph::Node>> gather_nd_args;
    gather_nd_args.push_back(gather_nd_1_slice_res);
    gather_nd_args.push_back(gather_nd_2_slice_res);
    gather_nd_args.push_back(gather_nd_3_slice_res);
    gather_nd_args.push_back(gather_nd_4_slice_res);
    auto concat = std::make_shared<op::Concat>(gather_nd_args, 0);

    return {concat};
}

shared_ptr<Node> op::GatherElements::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<GatherElements>(new_args.at(0), new_args.at(1), m_axis);
}

void op::GatherElements::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
    set_output_type(0, get_input_element_type(0), PartialShape::dynamic());

    if (is_dynamic())
    {
        return;
    }
}