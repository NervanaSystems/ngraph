// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

#include <memory>
using namespace std;
using namespace ngraph;

TEST(input_output, param_tensor)
{
    // Params have no arguments, so we can check that the value becomes a tensor output
    auto tv_tp = make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4});
    auto param = make_shared<op::Parameter>(tv_tp);

    ASSERT_EQ(param->get_outputs().size(), 1);
    for (size_t i = 0; i < param->get_outputs().size(); i++)
    {
        auto& output = param->get_outputs()[i];
        ASSERT_EQ(i, output.get_index());
        ASSERT_EQ(param, output.get_node());
    }

    ASSERT_EQ(*tv_tp, *param->get_outputs()[0].get_tensor_view()->get_tensor_view_type());
}

TEST(input_output, param_tuple)
{
    // Same as param_tensor, but for a tuple
    auto tv_tp_0 = make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4});
    auto tv_tp_1 = make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4, 6});
    auto tp_tp = make_shared<TupleType>(ValueTypes{tv_tp_0, tv_tp_1});
    auto param = make_shared<op::Parameter>(tp_tp);

    ASSERT_EQ(param->get_outputs().size(), 2);
    for (size_t i = 0; i < param->get_outputs().size(); i++)
    {
        auto& output = param->get_outputs()[i];
        ASSERT_EQ(i, output.get_index());
        ASSERT_EQ(param, output.get_node());
    }

    ASSERT_EQ(*tv_tp_0, *param->get_outputs()[0].get_tensor_view()->get_tensor_view_type());
    ASSERT_EQ(*tv_tp_1, *param->get_outputs()[1].get_tensor_view()->get_tensor_view_type());
}

TEST(input_output, simple_output)
{
    auto tv_tp_0 = make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 4});
    auto param_0 = make_shared<op::Parameter>(tv_tp_0);
    auto param_1 = make_shared<op::Parameter>(tv_tp_0);
    auto add = make_shared<op::Add>(param_0, param_1);

    // Sort the ops
    vector<shared_ptr<Node>> nodes;
    nodes.push_back(param_0);
    nodes.push_back(param_1);
    nodes.push_back(add);

    // At this point, the add should have each input associated with the output of the appropriate parameter
    ASSERT_EQ(1, add->get_outputs().size());
    auto& inputs = add->get_inputs();
    ASSERT_EQ(2, inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        auto& input = inputs[i];
        ASSERT_EQ(i, input.get_index());
        ASSERT_EQ(i, input.get_argno());
        ASSERT_EQ(0, input.get_arg_index());
        ASSERT_EQ(input.get_output().get_node(), add->get_arguments()[i]);
    }
}
