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
#include <memory>
#include <numeric>

#include "matmul.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/fused/stack.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Stack::type_info;

op::Stack::Stack(const OutputVector& args, int64_t axis)
    : FusedOp(OutputVector{args})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

op::Stack::Stack(const NodeVector& args, int64_t axis)
    : Stack(as_output_vector(args), axis)
{
}

shared_ptr<Node> op::Stack::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<Stack>(new_args, m_axis);
}

void op::Stack::pre_validate_and_infer_types()
{
    for (uint64_t i = 0; i < get_input_size(); ++i)
    {
        element::Type input_element_type = get_input_element_type(i);

        NODE_VALIDATION_CHECK(this,
                              input_element_type.is_dynamic() || input_element_type.is_real(),
                              "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                              input_element_type,
                              ").");
    }
}

NodeVector op::Stack::decompose_op() const
{
    auto axis = get_axis();
    std::vector<std::shared_ptr<ngraph::Node>> args;
    for (uint64_t i = 0; i < get_input_size(); ++i)
    {
        const PartialShape data_pshape = get_input_partial_shape(i);
        {
            auto data = input_value(i);
            auto data_shape = data.get_shape();
            axis = (axis < 0) ? axis + data_shape.size() + 1 : axis;

            data_shape.insert(data_shape.begin() + axis, 1);
            std::vector<size_t> input_order(data_shape.size() - 1);
            std::iota(std::begin(input_order), std::end(input_order), 0);
            args.push_back(
                std::make_shared<op::Reshape>(data, AxisVector(input_order), data_shape));
        }
    }
    auto concat = std::make_shared<op::Concat>(args, axis);
    return {concat};
}
