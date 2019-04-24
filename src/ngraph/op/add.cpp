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

#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

op::Add::Add(const shared_ptr<Node>& arg0, const shared_ptr<Node>& arg1)
    : BinaryElementwiseArithmetic("Add", arg0, arg1)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Add::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Add>(new_args.at(0), new_args.at(1));
}

std::vector<std::shared_ptr<op::Constant>> op::Add::as_constants() const
{
    //
    // For the time being we will only support vectors of int64 here, since that's all that's
    // needed for static shape propagation.
    //
    auto shape = input(0).get_shape();
    if (shape.size() != 1)
    {
        return {};
    }
    if (input(0).get_element_type() != element::i64)
    {
        return {};
    }

    NGRAPH_CHECK(input(0).get_shape() == input(1).get_shape());
    NGRAPH_CHECK(input(0).get_element_type() == input(1).get_element_type());

    auto left_node = dynamic_cast<op::Constant*>(input(0).get_source_output().get_node());
    auto right_node = dynamic_cast<op::Constant*>(input(1).get_source_output().get_node());

    if (left_node == nullptr || right_node == nullptr)
    {
        return {};
    }

    auto left_data = left_node->get_data_ptr<int64_t>();
    auto right_data = right_node->get_data_ptr<int64_t>();
    std::vector<int64_t> values(shape_size(shape));

    for (size_t i = 0; i < values.size(); i++)
    {
        values[i] = left_data[i] + right_data[i];
    }

    return {op::Constant::create(element::i64, shape, values)};
}

void op::Add::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);
    auto y = get_argument(1);

    adjoints.add_delta(x, delta);
    adjoints.add_delta(y, delta);
}

shared_ptr<Node> ngraph::operator+(const shared_ptr<Node> arg0, const shared_ptr<Node> arg1)
{
    return make_shared<op::Add>(arg0, arg1);
}
