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
#include "ngraph/op/fused/gather_element.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GatherElement::type_info;

op::GatherElement::GatherElement(const Output<Node>& arg1, const Output<Node>& arg2, int64_t axis)
    : FusedOp({arg1, arg2})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

NodeVector op::GatherElement::decompose_op() const
{
    auto axis = get_axis();
    auto data = input_value(0);
    auto indicate = input_value(1);

    auto x_shape = data.get_shape();
    auto indicate_shape = indicate.get_shape();

    auto reshape = [&](const Output<Node>& input, ngraph::Shape shape) {
        std::vector<size_t> input_order(input.get_shape().size());
        std::iota(std::begin(input_order), std::end(input_order), 0);
        std::shared_ptr<ngraph::Node> reshape =
            std::make_shared<op::Reshape>(input, ngraph::AxisVector(input_order), shape);
        return reshape;
    };

    for (auto& it : x_shape)
    {
        std::cout << it << " ";
    }
    std::cout << std::endl;
    for (auto& it : indicate_shape)
    {
        std::cout << it << " ";
    }
    std::cout << std::endl;
    std::vector<size_t> input_order(data.get_shape().size());
    std::iota(std::begin(input_order), std::end(input_order), 0);
    size_t axis_1 = x_shape[0];
    size_t axis_2 = 1;
    if (x_shape.size() > 1)
    {
        axis_2 = std::accumulate(
            std::begin(x_shape) + 1, std::end(x_shape), 1, std::multiplies<size_t>());
    }
    std::shared_ptr<ngraph::Node> x_reshape = std::make_shared<op::Reshape>(
        data, ngraph::AxisVector(input_order), ngraph::Shape{axis_1, axis_2});

    for (auto& it : x_reshape->get_shape())
    {
        std::cout << it << " ";
    }
    std::cout << std::endl << std::endl;
    // return reshape;
    // auto x_reshape = std::make_shared<ngraph::op::Reshape>(
    //    data, ngraph::AxisVector(x_order), ngraph::Shape{axis});
    auto result = std::make_shared<ngraph::op::EmbeddingLookup>(indicate, x_reshape);
    auto result_shape = result->get_shape();
    std::vector<size_t> out_shape(x_shape);
    out_shape[0] = result_shape[0];
     auto out =
        std::make_shared<ngraph::op::Reshape>(result, ngraph::AxisVector{0, 1}, out_shape);
    return {out};
}

shared_ptr<Node> op::GatherElement::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<GatherElement>(new_args.at(0), new_args.at(1), m_axis);
}

void op::GatherElement::pre_validate_and_infer_types()
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
