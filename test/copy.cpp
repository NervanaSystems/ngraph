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

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

template <typename OP>
bool check_unary()
{
    Shape shape{1};
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape)};

    auto node = make_shared<OP>(arg0);
    auto new_node = node->copy_with_new_args(new_args);

    return (nullptr != new_node) && (new_args == new_node->get_arguments());
}

template <typename OP>
bool check_binary()
{
    Shape shape{1};
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto arg1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape),
        make_shared<op::Parameter>(element::Float32::element_type(), shape)};

    auto node = make_shared<OP>(arg0, arg1);
    auto new_node = node->copy_with_new_args(new_args);

    return (nullptr != new_node) && (new_args == new_node->get_arguments());
}

TEST(copy, abs)
{
    ASSERT_TRUE(check_unary<op::Abs>());
}

TEST(copy, acos)
{
    ASSERT_TRUE(check_unary<op::Acos>());
}

TEST(copy, add)
{
    ASSERT_TRUE(check_binary<op::Add>());
}

TEST(copy, asin)
{
    ASSERT_TRUE(check_unary<op::Asin>());
}

TEST(copy, atan)
{
    ASSERT_TRUE(check_unary<op::Atan>());
}

TEST(copy, broadcast)
{
    Shape shape1{1};
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape1)};

    Shape shape{4, 1, 3};
    AxisSet axes{0, 2};

    auto node = make_shared<op::Broadcast>(arg0, shape, axes);
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::Broadcast>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
    ASSERT_TRUE(shape == node_cast->get_broadcast_shape());
    ASSERT_TRUE(axes == node_cast->get_broadcast_axes());
}

TEST(copy, ceiling)
{
    ASSERT_TRUE(check_unary<op::Ceiling>());
}

TEST(copy, concat)
{
    Shape shape{1};
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto arg1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape),
        make_shared<op::Parameter>(element::Float32::element_type(), shape)};
    size_t axis = 0;
    auto node = make_shared<op::Concat>(Nodes{arg0, arg1}, axis);
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::Concat>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
    ASSERT_TRUE(node_cast->get_concatenation_axis() == axis);
}

TEST(copy, parameterized_constant)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    // Create some tensors for input/output
    auto c = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{2, 2});
    copy_data(c, runtime::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());

    Shape shape{2, 2};
    auto cptv = dynamic_pointer_cast<ngraph::runtime::ParameterizedTensorView<element::Float32>>(c);
    ASSERT_NE(cptv, nullptr);
    auto node = make_shared<op::ParameterizedConstant<element::Float32>>(shape, cptv);
    auto new_node = node->copy_with_new_args(Nodes{});
    auto node_cast = dynamic_pointer_cast<op::ParameterizedConstant<element::Float32>>(new_node);
    ASSERT_NE(node_cast, nullptr);
    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(Nodes{} == new_node->get_arguments());
    ASSERT_TRUE(node_cast->get_value() == c);
    ASSERT_TRUE(node_cast->get_shape() == shape);
}

TEST(copy, constant)
{
    Shape shape{};
    vector<string> c{"2.4"};
    auto& et = element::Float32::element_type();
    auto node = make_shared<op::Constant>(et, shape, c);
    auto new_node = node->copy_with_new_args(Nodes{});
    auto node_cast = dynamic_pointer_cast<op::Constant>(new_node);
    ASSERT_NE(node_cast, nullptr);
    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(Nodes{} == new_node->get_arguments());
    ASSERT_TRUE(node_cast->get_value_strings() == c);
    ASSERT_TRUE(node_cast->get_shape() == shape);
    ASSERT_TRUE(node_cast->get_element_type() == et);
}

TEST(copy, convert)
{
    Shape shape;
    auto& et = element::Float64::element_type();
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape)};

    auto node = make_shared<op::Convert>(arg0, et);
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::Convert>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
    ASSERT_TRUE(et == node_cast->get_convert_element_type());
}

TEST(copy, cos)
{
    ASSERT_TRUE(check_unary<op::Cos>());
}

TEST(copy, cosh)
{
    ASSERT_TRUE(check_unary<op::Cosh>());
}

TEST(copy, divide)
{
    ASSERT_TRUE(check_binary<op::Divide>());
}

TEST(copy, dot)
{
    ASSERT_TRUE(check_binary<op::Dot>());
}

TEST(copy, equal)
{
    ASSERT_TRUE(check_binary<op::Equal>());
}

TEST(copy, exp)
{
    ASSERT_TRUE(check_unary<op::Exp>());
}

TEST(copy, floor)
{
    ASSERT_TRUE(check_unary<op::Floor>());
}

TEST(copy, FunctionCall)
{
    Shape shape{1};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>((A + B) * C, rt, op::Parameters{A, B, C});

    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto arg1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto arg2 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto node = make_shared<op::FunctionCall>(f, Nodes{arg0, arg1, arg2});

    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape),
        make_shared<op::Parameter>(element::Float32::element_type(), shape),
        make_shared<op::Parameter>(element::Float32::element_type(), shape)};
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::FunctionCall>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
    ASSERT_TRUE(node_cast->get_function() == f);
}

TEST(copy, GetTupleElement)
{
    Shape shape{1};
    size_t n = 0;
    auto tuple_type = make_shared<TupleType>(vector<shared_ptr<const ValueType>>{
        make_shared<TensorViewType>(element::Float32::element_type(), shape)});
    auto arg0 = make_shared<op::Parameter>(tuple_type);

    std::vector<std::shared_ptr<Node>> new_args{make_shared<op::Parameter>(tuple_type)};

    auto node = make_shared<op::GetTupleElement>(arg0, n);
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::GetTupleElement>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
    ASSERT_TRUE(node_cast->get_n() == n);
}

TEST(copy, greater_eq)
{
    ASSERT_TRUE(check_binary<op::GreaterEq>());
}

TEST(copy, greater)
{
    ASSERT_TRUE(check_binary<op::Greater>());
}

TEST(copy, less_eq)
{
    ASSERT_TRUE(check_binary<op::LessEq>());
}

TEST(copy, less)
{
    ASSERT_TRUE(check_binary<op::Less>());
}

TEST(copy, log)
{
    ASSERT_TRUE(check_unary<op::Log>());
}

TEST(copy, maximum)
{
    ASSERT_TRUE(check_binary<op::Maximum>());
}

TEST(copy, minimum)
{
    ASSERT_TRUE(check_binary<op::Minimum>());
}

TEST(copy, multiply)
{
    ASSERT_TRUE(check_binary<op::Multiply>());
}

TEST(copy, negative)
{
    ASSERT_TRUE(check_unary<op::Negative>());
}

TEST(copy, not_equal)
{
    ASSERT_TRUE(check_binary<op::NotEqual>());
}

TEST(copy, parameter)
{
    Shape shape{1};
    auto node = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto new_node = node->copy_with_new_args({});
    auto node_cast = dynamic_pointer_cast<op::Parameter>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_node->get_arguments().size() == 0);
    ASSERT_TRUE(node->get_value_type() == new_node->get_value_type());
}

TEST(copy, power)
{
    ASSERT_TRUE(check_binary<op::Power>());
}

TEST(copy, reduce)
{
    Shape scalar_shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), scalar_shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), scalar_shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), scalar_shape);
    auto f = make_shared<Function>(A + B, rt, op::Parameters{A, B});

    Shape shape{4, 3};
    AxisSet axes{1};
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto arg_init = make_shared<op::Parameter>(element::Float32::element_type(), scalar_shape);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape),
        make_shared<op::Parameter>(element::Float32::element_type(), scalar_shape)};

    auto node = make_shared<op::Reduce>(arg0, arg_init, f, axes);
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::Reduce>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
    ASSERT_TRUE(f == node_cast->get_function());
    ASSERT_TRUE(axes == node_cast->get_reduction_axes());
}

TEST(copy, remainder)
{
    ASSERT_TRUE(check_binary<op::Remainder>());
}

TEST(copy, reshape)
{
    Shape shape_in{2, 3, 4};
    AxisVector axes{0, 1, 2};
    Shape shape_out{6, 4};

    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape_in);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape_in)};

    auto node = make_shared<op::Reshape>(arg0, axes, shape_out);
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::Reshape>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
    ASSERT_TRUE(axes == node_cast->get_input_order());
    ASSERT_TRUE(shape_out == node_cast->get_output_shape());
}

TEST(copy, select)
{
    Shape shape{1};
    auto arg0 = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    auto arg1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto arg2 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Bool::element_type(), shape),
        make_shared<op::Parameter>(element::Float32::element_type(), shape),
        make_shared<op::Parameter>(element::Float32::element_type(), shape)};

    auto node = make_shared<op::Select>(arg0, arg1, arg2);
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::Select>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
}

TEST(copy, sign)
{
    ASSERT_TRUE(check_unary<op::Sign>());
}

TEST(copy, sin)
{
    ASSERT_TRUE(check_unary<op::Sin>());
}

TEST(copy, sinh)
{
    ASSERT_TRUE(check_unary<op::Sinh>());
}

TEST(copy, slice)
{
    Shape shape_in{2, 3, 4};
    Coordinate lower{0, 0, 0};
    Coordinate upper{2, 3, 4};
    Coordinate step{1, 1, 1};

    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape_in);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape_in)};

    auto node = make_shared<op::Slice>(arg0, lower, upper, step);
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::Slice>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
    ASSERT_TRUE(lower == node_cast->get_lower_bounds());
    ASSERT_TRUE(upper == node_cast->get_upper_bounds());
    ASSERT_TRUE(step == node_cast->get_step());
}

TEST(copy, subtract)
{
    ASSERT_TRUE(check_binary<op::Subtract>());
}

TEST(copy, sum)
{
    Shape shape{4, 3};
    AxisSet axes{1};
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape)};

    auto node = make_shared<op::Sum>(arg0, axes);
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::Sum>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
    ASSERT_TRUE(axes == node_cast->get_reduction_axes());
}

TEST(copy, tan)
{
    ASSERT_TRUE(check_unary<op::Tan>());
}

TEST(copy, tanh)
{
    ASSERT_TRUE(check_unary<op::Tanh>());
}

TEST(copy, tuple)
{
    Shape shape{1};
    auto arg0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto arg1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    std::vector<std::shared_ptr<Node>> new_args{
        make_shared<op::Parameter>(element::Float32::element_type(), shape),
        make_shared<op::Parameter>(element::Float32::element_type(), shape)};

    auto node = make_shared<op::Tuple>(Nodes{arg0, arg1});
    auto new_node = node->copy_with_new_args(new_args);
    auto node_cast = dynamic_pointer_cast<op::Tuple>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->get_arguments());
}
