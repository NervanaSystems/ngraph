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

#include "ngraph/pass/constant_folding.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(constant_folding, constant_squeeze)
{
    Shape shape_in{2, 4, 1};
    Shape shape_out{2, 4};
    Shape axes_shape{1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
    vector<int64_t> values_axes{2};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto squeeze = make_shared<op::Squeeze>(constant, constant_axes);
    auto f = make_shared<Function>(squeeze, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Squeeze>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), shape_out);

    auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_unsqueeze)
{
    Shape shape_in{2, 4};
    Shape shape_out{2, 4, 1, 1};
    Shape axes_shape{2};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
    vector<int64_t> values_axes{2, 3};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto unsqueeze = make_shared<op::Unsqueeze>(constant, constant_axes);
    auto f = make_shared<Function>(unsqueeze, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Unsqueeze>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), shape_out);

    auto values_out = new_const->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_reshape)
{
    Shape shape_in{2, 4};
    Shape shape_out{2, 4, 1};

    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto reshape = make_shared<op::Reshape>(constant, AxisVector{0, 1}, shape_out);
    auto f = make_shared<Function>(reshape, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<float>();

    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_reshape_permute)
{
    Shape shape_in{2, 4};
    Shape shape_out{4, 2};

    vector<double> values_in{0, 1, 2, 3, 4, 5, 6, 7};
    auto constant = make_shared<op::Constant>(element::f64, shape_in, values_in);
    auto reshape = make_shared<op::Reshape>(constant, AxisVector{1, 0}, shape_out);
    auto f = make_shared<Function>(reshape, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<double>();

    vector<double> values_permute{0, 4, 1, 5, 2, 6, 3, 7};
    ASSERT_TRUE(test::all_close_f(values_permute, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_broadcast)
{
    Shape shape_in{2};
    Shape shape_out{2, 4};

    vector<int> values_in{0, 1};
    auto constant = make_shared<op::Constant>(element::i32, shape_in, values_in);
    auto broadcast = make_shared<op::Broadcast>(constant, shape_out, AxisSet{1});
    auto f = make_shared<Function>(broadcast, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{0, 0, 0, 0, 1, 1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_dyn_broadcast)
{
    vector<int32_t> values_in{0, 1};
    auto constant_in = make_shared<op::Constant>(element::i32, Shape{2}, values_in);
    vector<int64_t> shape_in{2, 4};
    auto constant_shape = make_shared<op::Constant>(element::i64, Shape{2}, shape_in);
    vector<int64_t> axes_in{1};
    auto constant_axes = make_shared<op::Constant>(element::i64, Shape{1}, axes_in);
    auto dyn_broadcast = make_shared<op::DynBroadcast>(constant_in, constant_shape, constant_axes);
    auto f = make_shared<Function>(dyn_broadcast, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::DynBroadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{0, 0, 0, 0, 1, 1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_broadcast_v1)
{
    vector<int32_t> values_in{0, 1};
    auto constant_in = make_shared<op::Constant>(element::i32, Shape{2}, values_in);
    vector<int64_t> shape_in{2, 4};
    auto constant_shape = make_shared<op::Constant>(element::i64, Shape{2}, shape_in);
    vector<int64_t> axes_in{0};
    auto constant_axes = make_shared<op::Constant>(element::i64, Shape{1}, axes_in);
    auto broadcast_v1 = make_shared<op::v1::Broadcast>(constant_in, constant_shape, constant_axes);
    auto f = make_shared<Function>(broadcast_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{0, 0, 0, 0, 1, 1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_broadcast_v1_with_target_shape)
{
    vector<int32_t> values_in{1};
    auto constant_in = make_shared<op::Constant>(element::i32, Shape{1, 1, 1, 1}, values_in);
    vector<int64_t> shape_in{1, 3, 1, 1};
    auto target_shape = make_shared<op::Constant>(element::i64, Shape{4}, shape_in);
    auto broadcast_v1 = make_shared<op::v1::Broadcast>(constant_in, target_shape);
    auto f = make_shared<Function>(broadcast_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_broadcast_v1_numpy)
{
    vector<int32_t> values_in{0, 1};
    auto constant_in = make_shared<op::Constant>(element::i32, Shape{2}, values_in);
    vector<int64_t> shape_in{4, 2};
    auto constant_shape = make_shared<op::Constant>(element::i64, Shape{2}, shape_in);
    auto broadcast_v1 = make_shared<op::v1::Broadcast>(constant_in, constant_shape);
    auto f = make_shared<Function>(broadcast_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Broadcast>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{0, 1, 0, 1, 0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_pad_exterior)
{
    Shape shape_in{2};

    vector<int> values_in{777, 888};
    auto constant = make_shared<op::Constant>(element::i32, shape_in, values_in);
    auto pad_value = make_shared<op::Constant>(element::i32, Shape{}, vector<int>{111});

    CoordinateDiff padding_below{1};
    CoordinateDiff padding_above{2};

    auto broadcast = make_shared<op::Pad>(constant, pad_value, padding_below, padding_above);
    auto f = make_shared<Function>(broadcast, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Pad>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> padded_values{111, 777, 888, 111, 111};
    ASSERT_EQ(padded_values, values_out);
}

template <typename T>
static std::vector<T> get_result_constant(std::shared_ptr<Function> f, size_t pos)
{
    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(pos)->get_argument(0));
    return new_const->get_vector<T>();
}

TEST(constant_folding, constant_unary_binary)
{
    vector<int> values_a{1, 2, 3, 4};
    vector<int> values_b{1, 2, 3, 4};
    vector<int> values_c{-1, -1, -1, -1};
    vector<int> values_d{1, 4, 9, 16};
    vector<int> values_e{5, 6};
    vector<int> values_f{0, 10};
    vector<int> values_g{1, 4};
    vector<char> values_h{0, 0, 1, 1};
    vector<char> values_i{0, 1};
    auto a = make_shared<op::Constant>(element::i32, Shape{2, 2}, values_a);
    auto b = make_shared<op::Constant>(element::i32, Shape{2, 2}, values_b);
    auto c = make_shared<op::Constant>(element::i32, Shape{2, 2}, values_c);
    auto d = make_shared<op::Constant>(element::i32, Shape{2, 2}, values_d);
    auto e = make_shared<op::Constant>(element::i32, Shape{2}, values_e);
    auto f = make_shared<op::Constant>(element::i32, Shape{2}, values_f);
    auto g = make_shared<op::Constant>(element::i32, Shape{2}, values_g);
    auto h = make_shared<op::Constant>(element::boolean, Shape{2, 2}, values_h);
    auto i = make_shared<op::Constant>(element::boolean, Shape{2}, values_i);

    auto add = a + b;
    auto sub = a - b;
    auto mul = a * b;
    auto divn = a / b;
    auto pow = make_shared<op::Power>(a, b);
    auto min = make_shared<op::Minimum>(c, a);
    auto max = make_shared<op::Maximum>(a, c);
    auto absn = make_shared<op::Abs>(c);
    auto neg = make_shared<op::Negative>(c);
    auto sqrt = make_shared<op::Sqrt>(d);
    auto add_autob_numpy = make_shared<op::Add>(a, e, op::AutoBroadcastType::NUMPY);
    auto sub_autob_numpy = make_shared<op::Subtract>(a, e, op::AutoBroadcastType::NUMPY);
    auto mul_autob_numpy = make_shared<op::Multiply>(a, e, op::AutoBroadcastType::NUMPY);
    auto div_autob_numpy = make_shared<op::Divide>(a, g, op::AutoBroadcastType::NUMPY);
    auto pow_autob_numpy = make_shared<op::Power>(a, g, op::AutoBroadcastType::NUMPY);
    auto min_autob_numpy = make_shared<op::Minimum>(a, f, op::AutoBroadcastType::NUMPY);
    auto max_autob_numpy = make_shared<op::Maximum>(a, f, op::AutoBroadcastType::NUMPY);
    auto equal_autob_numpy = make_shared<op::Equal>(a, g, op::AutoBroadcastType::NUMPY);
    auto not_equal_autob_numpy = make_shared<op::NotEqual>(a, g, op::AutoBroadcastType::NUMPY);
    auto greater_autob_numpy = make_shared<op::Greater>(a, g, op::AutoBroadcastType::NUMPY);
    auto greater_eq_autob_numpy = make_shared<op::GreaterEq>(a, g, op::AutoBroadcastType::NUMPY);
    auto less_autob_numpy = make_shared<op::Less>(a, g, op::AutoBroadcastType::NUMPY);
    auto less_eq_autob_numpy = make_shared<op::LessEq>(a, g, op::AutoBroadcastType::NUMPY);
    auto logical_and_autob_numpy = make_shared<op::And>(h, i, op::AutoBroadcastType::NUMPY);
    auto logical_or_autob_numpy = make_shared<op::Or>(h, i, op::AutoBroadcastType::NUMPY);
    auto logical_xor_autob_numpy = make_shared<op::Xor>(h, i, op::AutoBroadcastType::NUMPY);

    auto neg_sqrt = make_shared<op::Sqrt>(c);

    auto func = make_shared<Function>(NodeVector{add,
                                                 sub,
                                                 mul,
                                                 divn,
                                                 pow,
                                                 min,
                                                 max,
                                                 absn,
                                                 neg,
                                                 sqrt,
                                                 add_autob_numpy,
                                                 sub_autob_numpy,
                                                 mul_autob_numpy,
                                                 div_autob_numpy,
                                                 pow_autob_numpy,
                                                 min_autob_numpy,
                                                 max_autob_numpy,
                                                 equal_autob_numpy,
                                                 not_equal_autob_numpy,
                                                 greater_autob_numpy,
                                                 greater_eq_autob_numpy,
                                                 less_autob_numpy,
                                                 less_eq_autob_numpy,
                                                 logical_and_autob_numpy,
                                                 logical_or_autob_numpy,
                                                 logical_xor_autob_numpy},
                                      ParameterVector{});
    auto func_error = make_shared<Function>(NodeVector{neg_sqrt}, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(func);

    // expected values
    vector<int> add_expected{2, 4, 6, 8};
    vector<int> sub_expected{0, 0, 0, 0};
    vector<int> mul_expected{1, 4, 9, 16};
    vector<int> div_expected{1, 1, 1, 1};
    vector<int> pow_expected{1, 4, 27, 256};
    vector<int> min_expected{-1, -1, -1, -1};
    vector<int> max_expected{1, 2, 3, 4};
    vector<int> abs_neg_expected{1, 1, 1, 1};
    vector<int> sqrt_expected{1, 2, 3, 4};
    vector<int> add_autob_numpy_expected{6, 8, 8, 10};
    vector<int> sub_autob_numpy_expected{-4, -4, -2, -2};
    vector<int> mul_autob_numpy_expected{5, 12, 15, 24};
    vector<int> div_autob_numpy_expected{1, 0, 3, 1};
    vector<int> pow_autob_numpy_expected{1, 16, 3, 256};
    vector<int> min_autob_numpy_expected{0, 2, 0, 4};
    vector<int> max_autob_numpy_expected{1, 10, 3, 10};
    vector<char> equal_autob_numpy_expected{1, 0, 0, 1};
    vector<char> not_equal_autob_numpy_expected{0, 1, 1, 0};
    vector<char> greater_autob_numpy_expected{0, 0, 1, 0};
    vector<char> greater_eq_autob_numpy_expected{1, 0, 1, 1};
    vector<char> less_autob_numpy_expected{0, 1, 0, 0};
    vector<char> less_eq_autob_numpy_expected{1, 1, 0, 1};
    vector<char> logical_and_autob_numpy_expected{0, 0, 0, 1};
    vector<char> logical_or_autob_numpy_expected{0, 1, 1, 1};
    vector<char> logical_xor_autob_numpy_expected{0, 1, 1, 0};

    ASSERT_EQ(get_result_constant<int>(func, 0), add_expected);
    ASSERT_EQ(get_result_constant<int>(func, 1), sub_expected);
    ASSERT_EQ(get_result_constant<int>(func, 2), mul_expected);
    ASSERT_EQ(get_result_constant<int>(func, 3), div_expected);
    ASSERT_EQ(get_result_constant<int>(func, 4), pow_expected);
    ASSERT_EQ(get_result_constant<int>(func, 5), min_expected);
    ASSERT_EQ(get_result_constant<int>(func, 6), max_expected);
    ASSERT_EQ(get_result_constant<int>(func, 7), abs_neg_expected);
    ASSERT_EQ(get_result_constant<int>(func, 8), abs_neg_expected);
    ASSERT_EQ(get_result_constant<int>(func, 9), sqrt_expected);
    ASSERT_EQ(get_result_constant<int>(func, 10), add_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 11), sub_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 12), mul_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 13), div_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 14), pow_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 15), min_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<int>(func, 16), max_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 17), equal_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 18), not_equal_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 19), greater_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 20), greater_eq_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 21), less_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 22), less_eq_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 23), logical_and_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 24), logical_or_autob_numpy_expected);
    ASSERT_EQ(get_result_constant<char>(func, 25), logical_xor_autob_numpy_expected);
    ASSERT_ANY_THROW(pass_manager.run_passes(func_error));
}

TEST(constant_folding, const_dequantize)
{
    Shape input_shape{12};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto quant_type = element::u8;
    auto output_type = element::f32;
    typedef float output_c_type;

    vector<uint8_t> values_in{1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
    auto constant = op::Constant::create(quant_type, input_shape, values_in);
    auto scale = op::Constant::create(output_type, scale_offset_shape, {2});
    auto offset = op::Constant::create(quant_type, scale_offset_shape, {1});
    auto dequantize =
        make_shared<op::Dequantize>(constant, scale, offset, output_type, quantization_axes);
    auto f = make_shared<Function>(dequantize, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Dequantize>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<output_c_type>();

    vector<output_c_type> values_dequantize{0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12};
    ASSERT_EQ(values_dequantize, values_out);
}

TEST(constant_folding, const_quantize)
{
    Shape input_shape{12};
    Shape scale_offset_shape;
    AxisSet quantization_axes;

    auto quant_type = element::u8;
    auto output_type = element::u8;
    typedef uint8_t output_c_type;

    vector<float> values_in{1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0};
    auto constant = op::Constant::create(element::f32, input_shape, values_in);
    auto scale = op::Constant::create(element::f32, scale_offset_shape, {2});
    auto offset = op::Constant::create(quant_type, scale_offset_shape, {1});
    auto mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY;
    auto quantize =
        make_shared<op::Quantize>(constant, scale, offset, output_type, quantization_axes, mode);
    auto f = make_shared<Function>(quantize, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Quantize>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<output_c_type>();

    vector<output_c_type> values_quantize{2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5};
    ASSERT_EQ(values_quantize, values_out);
}

TEST(constant_folding, const_convert)
{
    Shape input_shape{3, 4};

    vector<int32_t> values_in{1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
    auto constant = op::Constant::create(element::f32, input_shape, values_in);
    auto convert = make_shared<op::Convert>(constant, element::u64);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Convert>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_output_element_type(0), element::u64);
    auto values_out = new_const->get_vector<uint64_t>();

    vector<uint64_t> values_expected{1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, shape_of)
{
    Shape input_shape{3, 4, 0, 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::ShapeOf>(param);
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::ShapeOf>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_output_element_type(0), element::i64);
    auto values_out = new_const->get_vector<int64_t>();

    ASSERT_EQ((vector<int64_t>{3, 4, 0, 22, 608, 909, 3}), values_out);
}

// A bit of an unusual case here: constant folding will not succeed on ShapeOf
// if the argument doesn't have dynamic shape. We want to make sure it fails
// gracefully, leaving the ShapeOf op in place.
TEST(constant_folding, shape_of_dynamic)
{
    PartialShape input_shape{3, 4, Dimension::dynamic(), 22, 608, 909, 3};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::ShapeOf>(param);
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 0);

    auto result_as_shape_of = as_type_ptr<op::ShapeOf>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(result_as_shape_of);
    ASSERT_EQ(result_as_shape_of->get_output_shape(0), Shape{7});
}

// Similar to shape_of_dynamic above but here even the rank is dynamic.
TEST(constant_folding, shape_of_rank_dynamic)
{
    PartialShape input_shape{PartialShape::dynamic()};

    auto param = make_shared<op::Parameter>(element::boolean, input_shape);
    auto shape_of = make_shared<op::ShapeOf>(param);
    auto f = make_shared<Function>(shape_of, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::ShapeOf>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 0);

    auto result_as_shape_of = as_type_ptr<op::ShapeOf>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(result_as_shape_of);
    ASSERT_TRUE(result_as_shape_of->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic()}));
}

TEST(constant_folding, const_reverse)
{
    Shape input_shape{3, 3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    auto convert = make_shared<op::Reverse>(constant, AxisSet{1});
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Reverse>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{3, 2, 1, 6, 5, 4, 9, 8, 7};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_product)
{
    Shape input_shape{3, 3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    auto convert = make_shared<op::Product>(constant, AxisSet{1});
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Product>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 120, 504};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduceprod)
{
    Shape input_shape{3, 3};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceProd>(constant, constant_axes);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceProd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 120, 504};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduceprod_keepdims)
{
    Shape input_shape{3, 3};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceProd>(constant, constant_axes, true);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceProd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 120, 504};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_sum)
{
    Shape input_shape{3, 3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    auto convert = make_shared<op::Sum>(constant, AxisSet{1});
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Sum>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 15, 24};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducesum)
{
    Shape input_shape{3, 3};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceSum>(constant, constant_axes);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceSum>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 15, 24};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducesum_keepdims)
{
    Shape input_shape{3, 3};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceSum>(constant, constant_axes, true);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceSum>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 15, 24};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_max)
{
    Shape input_shape{3, 3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    auto convert = make_shared<op::Max>(constant, AxisSet{1});
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Max>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{3, 6, 9};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemax)
{
    Shape input_shape{3, 2};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMax>(constant, constant_axes);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMax>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 4, 6};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemax_keepdims)
{
    Shape input_shape{3, 2};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMax>(constant, constant_axes, true);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMax>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 4, 6};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_min)
{
    Shape input_shape{3, 3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    auto convert = make_shared<op::Min>(constant, AxisSet{1});
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Min>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 4, 7};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemin)
{
    Shape input_shape{3, 2};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMin>(constant, constant_axes);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMin>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 3, 5};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemin_keepdims)
{
    Shape input_shape{3, 2};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMin>(constant, constant_axes, true);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMin>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 3, 5};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemean)
{
    Shape input_shape{3, 3};
    Shape output_shape{3};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMean>(constant, constant_axes);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMean>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 5, 8};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reducemean_keepdims)
{
    Shape input_shape{3, 3};
    Shape output_shape{3, 1};

    vector<int32_t> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto constant = op::Constant::create(element::i32, input_shape, values_in);
    Shape axes_shape{1};
    vector<int32_t> values_axes{1};
    auto constant_axes = op::Constant::create(element::i64, axes_shape, values_axes);
    auto convert = make_shared<op::v1::ReduceMean>(constant, constant_axes, true);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceMean>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    ASSERT_EQ(new_const->get_shape(), output_shape);

    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{2, 5, 8};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_all)
{
    Shape input_shape{3, 3};

    vector<char> values_in{0, 1, 1, 0, 1, 0, 1, 1, 1};
    auto constant = op::Constant::create(element::boolean, input_shape, values_in);
    auto convert = make_shared<op::All>(constant, AxisSet{1});
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::All>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_and__no_keepdims)
{
    const Shape input_shape{3, 3};

    const vector<char> values_in{0, 1, 1, 0, 1, 0, 1, 1, 1};
    const auto data = op::Constant::create(element::boolean, input_shape, values_in);
    const auto axes = op::Constant::create(element::i64, {1}, {1});
    const auto convert = make_shared<op::v1::ReduceLogicalAnd>(data, axes, false);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);

    const Shape expected_out_shape{3};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_and__keepdims)
{
    const Shape input_shape{3, 3};

    const vector<char> values_in{0, 1, 1, 0, 1, 0, 1, 1, 1};
    const auto data = op::Constant::create(element::boolean, input_shape, values_in);
    const auto axes = op::Constant::create(element::i64, {1}, {1});
    const auto convert = make_shared<op::v1::ReduceLogicalAnd>(data, axes, true);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);

    // the output shape is expected to have 'ones' at the positions specified in the reduction axes
    // in case the keep_dims attribute of ReduceLogicalAnd is set to true
    const Shape expected_out_shape{3, 1};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_and__keepdims_3d)
{
    const Shape input_shape{2, 2, 2};

    const vector<char> values_in{1, 1, 0, 0, 1, 0, 0, 1};
    const auto data = op::Constant::create(element::boolean, input_shape, values_in);
    const auto axes = op::Constant::create(element::i64, {2}, {0, 2});
    const auto convert = make_shared<op::v1::ReduceLogicalAnd>(data, axes, true);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);

    const Shape expected_out_shape{1, 2, 1};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_any)
{
    Shape input_shape{3, 3};

    vector<char> values_in{1, 0, 0, 1, 0, 1, 0, 0, 0};
    auto constant = op::Constant::create(element::boolean, input_shape, values_in);
    auto convert = make_shared<op::Any>(constant, AxisSet{1});
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Any>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_reduce_logical_or__no_keepdims)
{
    const Shape input_shape{3, 3};

    const vector<char> values_in{1, 0, 0, 1, 0, 1, 0, 0, 0};
    const auto data = op::Constant::create(element::boolean, input_shape, values_in);
    const auto axes = op::Constant::create(element::i64, {1}, {1});
    const auto convert = make_shared<op::v1::ReduceLogicalOr>(data, axes, false);
    auto f = make_shared<Function>(convert, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::ReduceLogicalAnd>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    const auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);

    const Shape expected_out_shape{3};
    ASSERT_EQ(new_const->get_shape(), expected_out_shape);

    const auto values_out = new_const->get_vector<char>();

    const vector<char> values_expected{1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_concat)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 = op::Constant::create(element::i32, Shape{2, 1}, vector<int32_t>{7, 8});
    auto concat = make_shared<op::Concat>(NodeVector{constant0, constant1}, 1);
    auto f = make_shared<Function>(concat, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Concat>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{1, 2, 3, 7, 4, 5, 6, 8};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_not)
{
    auto constant =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<char>{0, 1, 0, 0, 1, 1});
    auto logical_not = make_shared<op::Not>(constant);
    auto f = make_shared<Function>(logical_not, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Not>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 0, 1, 1, 0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_equal)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 2, 3, 5, 6});
    auto eq = make_shared<op::Equal>(constant0, constant1);
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Equal>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 1, 0, 0, 1, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_not_equal)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 2, 3, 5, 6});
    auto eq = make_shared<op::NotEqual>(constant0, constant1);
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::NotEqual>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 0, 1, 1, 0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_greater)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    auto eq = make_shared<op::Greater>(constant0, constant1);
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Greater>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 0, 1, 0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_greater_eq)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    auto eq = make_shared<op::GreaterEq>(constant0, constant1);
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::GreaterEq>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 1, 1, 0, 1, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_less)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    auto eq = make_shared<op::Less>(constant0, constant1);
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Less>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 0, 0, 1, 0, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_less_eq)
{
    auto constant0 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{1, 2, 3, 4, 5, 6});
    auto constant1 =
        op::Constant::create(element::i32, Shape{2, 3}, vector<int32_t>{2, 2, 2, 5, 5, 5});
    auto eq = make_shared<op::LessEq>(constant0, constant1);
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::LessEq>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{1, 1, 0, 1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_and)
{
    auto constant0 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 0, 1, 0, 1, 1});
    auto constant1 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 1, 1, 1, 0, 1});
    auto eq = make_shared<op::And>(constant0, constant1);
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::And>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 0, 1, 0, 0, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_or)
{
    auto constant0 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 0, 1, 0, 1, 1});
    auto constant1 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 1, 1, 1, 0, 1});
    auto eq = make_shared<op::Or>(constant0, constant1);
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Or>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 1, 1, 1, 1, 1};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_xor)
{
    auto constant0 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 0, 1, 0, 1, 1});
    auto constant1 =
        op::Constant::create(element::boolean, Shape{2, 3}, vector<int32_t>{0, 1, 1, 1, 0, 1});
    auto eq = make_shared<op::Xor>(constant0, constant1);
    auto f = make_shared<Function>(eq, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Xor>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<char>();

    vector<char> values_expected{0, 1, 0, 1, 1, 0};

    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, const_ceiling)
{
    auto constant = op::Constant::create(
        element::f32, Shape{2, 3}, vector<float>{0.0f, 0.1f, -0.1f, -2.5f, 2.5f, 3.0f});
    auto ceil = make_shared<op::Ceiling>(constant);
    auto f = make_shared<Function>(ceil, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Ceiling>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{0.0f, 1.0f, 0.0f, -2.0f, 3.0f, 3.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_floor)
{
    auto constant = op::Constant::create(
        element::f32, Shape{2, 3}, vector<float>{0.0f, 0.1f, -0.1f, -2.5f, 2.5f, 3.0f});
    auto floor = make_shared<op::Floor>(constant);
    auto f = make_shared<Function>(floor, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Floor>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{0.0f, 0.0f, -1.0f, -3.0f, 2.0f, 3.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather)
{
    auto constant_data = op::Constant::create(
        element::f32,
        Shape{2, 5},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    auto constant_indices =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 2});
    size_t gather_axis = 1;
    auto gather = make_shared<op::v0::Gather>(constant_data, constant_indices, gather_axis);
    auto f = make_shared<Function>(gather, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{1.0f, 4.0f, 3.0f, 3.0f, 6.0f, 9.0f, 8.0f, 8.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1)
{
    auto constant_data = op::Constant::create(
        element::f32,
        Shape{2, 5},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    auto constant_indices =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 2});
    auto constant_axis = op::Constant::create(element::i64, Shape{1}, vector<int64_t>{1});
    auto gather = make_shared<op::v1::Gather>(constant_data, constant_indices, constant_axis);
    auto f = make_shared<Function>(gather, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{1.0f, 4.0f, 3.0f, 3.0f, 6.0f, 9.0f, 8.0f, 8.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_gather_v1_scalar)
{
    auto constant_data = op::Constant::create(
        element::f32,
        Shape{2, 5},
        vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
    auto constant_indices =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{0, 3, 2, 2});
    auto constant_axis = op::Constant::create(element::i64, Shape{}, vector<int64_t>{1});
    auto gather = make_shared<op::v1::Gather>(constant_data, constant_indices, constant_axis);
    auto f = make_shared<Function>(gather, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{1.0f, 4.0f, 3.0f, 3.0f, 6.0f, 9.0f, 8.0f, 8.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, const_slice)
{
    Shape shape_in{16};

    vector<int> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto constant = make_shared<op::Constant>(element::i32, shape_in, values_in);
    auto slice = make_shared<op::Slice>(constant, Coordinate{2}, Coordinate{15}, Strides{3});

    auto f = make_shared<Function>(slice, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Slice>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> sliced_values{3, 6, 9, 12, 15};
    ASSERT_EQ(sliced_values, values_out);
}

TEST(constant_folding, const_dyn_slice)
{
    Shape shape_in{16};

    vector<int> values_in{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto constant_data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_lb{2};
    auto constant_lb = make_shared<op::Constant>(element::i64, Shape{1}, values_lb);
    vector<int> values_ub{15};
    auto constant_ub = make_shared<op::Constant>(element::i64, Shape{1}, values_ub);
    vector<int> values_strides{3};
    auto constant_strides = make_shared<op::Constant>(element::i64, Shape{1}, values_strides);
    auto dyn_slice = make_shared<op::DynSlice>(constant_data,
                                               constant_lb,
                                               constant_ub,
                                               constant_strides,
                                               AxisSet{},
                                               AxisSet{},
                                               AxisSet{},
                                               AxisSet{},
                                               AxisSet{});

    auto f = make_shared<Function>(dyn_slice, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::DynSlice>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> sliced_values{3, 6, 9, 12, 15};
    ASSERT_EQ(sliced_values, values_out);
}

TEST(constant_folding, constant_dyn_reshape)
{
    Shape shape_in{2, 4};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};

    Shape shape_shape{3};
    vector<int64_t> values_shape{2, 4, 1};

    auto constant_in = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto constant_shape = make_shared<op::Constant>(element::i64, shape_shape, values_shape);
    auto dyn_reshape = make_shared<op::v1::Reshape>(constant_in, constant_shape, false);
    auto f = make_shared<Function>(dyn_reshape, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<float>();

    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_dyn_reshape_shape_not_originally_constant)
{
    Shape shape_in{2, 4};
    vector<float> values_in{0, 1, 2, 3, 4, 5, 6, 7};

    Shape shape_shape{3};
    // We're going to add these two together elementwise to get {2, 4, 1}.
    // This means that when ConstantFolding starts, v1::Reshape will not yet
    // have static output shape. But by the time the Add op is folded, the
    // v1::Reshape's shape should be inferrable.
    vector<int64_t> values_shape_a{1, 3, 0};
    vector<int64_t> values_shape_b{1, 1, 1};

    auto constant_in = make_shared<op::Constant>(element::f32, shape_in, values_in);
    auto constant_shape_a = make_shared<op::Constant>(element::i64, shape_shape, values_shape_a);
    auto constant_shape_b = make_shared<op::Constant>(element::i64, shape_shape, values_shape_b);
    auto dyn_reshape =
        make_shared<op::v1::Reshape>(constant_in, constant_shape_a + constant_shape_b, false);
    auto f = make_shared<Function>(dyn_reshape, ParameterVector{});

    ASSERT_TRUE(dyn_reshape->get_output_partial_shape(0).is_dynamic());

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Reshape>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<float>();

    ASSERT_TRUE(test::all_close_f(values_in, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

TEST(constant_folding, constant_transpose)
{
    Shape shape_in{2, 4};
    vector<double> values_in{0, 1, 2, 3, 4, 5, 6, 7};

    Shape shape_perm{2};
    vector<int64_t> values_perm{1, 0};

    auto constant_in = make_shared<op::Constant>(element::f64, shape_in, values_in);
    auto constant_perm = make_shared<op::Constant>(element::i64, shape_perm, values_perm);
    auto transpose = make_shared<op::Transpose>(constant_in, constant_perm);
    auto f = make_shared<Function>(transpose, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Transpose>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<double>();

    vector<double> values_permute{0, 4, 1, 5, 2, 6, 3, 7};
    ASSERT_TRUE(test::all_close_f(values_permute, values_out, MIN_FLOAT_TOLERANCE_BITS));
}

void range_test_check(const vector<double>& values_out, const vector<double>& values_expected)
{
    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

void range_test_check(const vector<float>& values_out, const vector<float>& values_expected)
{
    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type
    range_test_check(const vector<T>& values_out, const vector<T>& values_expected)
{
    ASSERT_EQ(values_out, values_expected);
}

template <typename T>
void range_test(T start, T stop, T step, const vector<T>& values_expected)
{
    vector<T> values_start{start};
    vector<T> values_stop{stop};
    vector<T> values_step{step};

    auto constant_start = make_shared<op::Constant>(element::from<T>(), Shape{}, values_start);
    auto constant_stop = make_shared<op::Constant>(element::from<T>(), Shape{}, values_stop);
    auto constant_step = make_shared<op::Constant>(element::from<T>(), Shape{}, values_step);
    auto range = make_shared<op::Range>(constant_start, constant_stop, constant_step);
    auto f = make_shared<Function>(range, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Range>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);

    auto values_out = new_const->template get_vector<T>();

    range_test_check(values_out, values_expected);
}

TEST(constant_folding, constant_range)
{
    range_test<int8_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<int32_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<int64_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<uint64_t>(5, 12, 2, {5, 7, 9, 11});
    range_test<double>(5, 12, 2, {5, 7, 9, 11});
    range_test<float>(5, 12, 2, {5, 7, 9, 11});

    range_test<int32_t>(5, 12, -2, {});
    range_test<float>(12, 4, -2, {12, 10, 8, 6});
}

TEST(constant_folding, constant_select)
{
    Shape shape{2, 4};
    vector<char> values_selection{0, 1, 1, 0, 1, 0, 0, 1};
    vector<int64_t> values_t{2, 4, 6, 8, 10, 12, 14, 16};
    vector<int64_t> values_f{1, 3, 5, 7, 9, 11, 13, 15};

    auto constant_selection = make_shared<op::Constant>(element::boolean, shape, values_selection);
    auto constant_t = make_shared<op::Constant>(element::i64, shape, values_t);
    auto constant_f = make_shared<op::Constant>(element::i64, shape, values_f);
    auto select = make_shared<op::Select>(constant_selection, constant_t, constant_f);
    auto f = make_shared<Function>(select, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Select>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int64_t>();

    vector<int64_t> values_expected{1, 4, 6, 7, 10, 11, 13, 16};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_v1_select)
{
    Shape shape{2, 4};
    vector<char> values_selection{0, 1, 1, 0};
    vector<int64_t> values_t{1, 2, 3, 4};
    vector<int64_t> values_f{11, 12, 13, 14, 15, 16, 17, 18};

    auto constant_selection =
        make_shared<op::Constant>(element::boolean, Shape{4}, values_selection);
    auto constant_t = make_shared<op::Constant>(element::i64, Shape{4}, values_t);
    auto constant_f = make_shared<op::Constant>(element::i64, Shape{2, 4}, values_f);
    auto select = make_shared<op::v1::Select>(constant_selection, constant_t, constant_f);
    auto f = make_shared<Function>(select, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Select>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int64_t>();

    vector<int64_t> values_expected{11, 2, 3, 14, 15, 2, 3, 18};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_v1_split)
{
    vector<float> data{.1f, .2f, .3f, .4f, .5f, .6f};
    const auto const_data = op::Constant::create(element::f32, Shape{data.size()}, data);
    const auto const_axis = op::Constant::create(element::i64, Shape{}, {0});
    const auto num_splits = 3;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Function>(split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), num_splits);

    auto res1 = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    auto res2 = as_type_ptr<op::Constant>(f->get_results().at(1)->get_argument(0));
    auto res3 = as_type_ptr<op::Constant>(f->get_results().at(2)->get_argument(0));
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);

    auto res1_values = res1->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin(), data.begin() + 2), res1_values));
    auto res2_values = res2->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin() + 2, data.begin() + 4), res2_values));
    auto res3_values = res3->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin() + 4, data.end()), res3_values));
}

TEST(constant_folding, constant_v1_split_specialized)
{
    vector<float> data{.1f, .2f, .3f, .4f, .5f, .6f};
    const auto const_data = op::Constant::create(element::f32, Shape{data.size()}, data);
    const auto const_axis = op::Constant::create(element::i64, Shape{}, {0});
    const auto num_splits = 3;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Function>(split_v1->outputs(), ParameterVector{});

    auto specialized_function = ::ngraph::specialize_function(
        std::const_pointer_cast<ngraph::Function>(f), {}, {}, {}, true, true);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), num_splits);

    auto res1 = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    auto res2 = as_type_ptr<op::Constant>(f->get_results().at(1)->get_argument(0));
    auto res3 = as_type_ptr<op::Constant>(f->get_results().at(2)->get_argument(0));
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);

    auto res1_values = res1->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin(), data.begin() + 2), res1_values));
    auto res2_values = res2->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin() + 2, data.begin() + 4), res2_values));
    auto res3_values = res3->get_vector<float>();
    ASSERT_TRUE(test::all_close_f(vector<float>(data.begin() + 4, data.end()), res3_values));
}

TEST(constant_folding, constant_v1_split_axis_1_4_splits)
{
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = op::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = op::Constant::create(element::i64, Shape{}, {1});
    const auto num_splits = 4;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Function>(split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), num_splits);

    auto res1 = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    auto res2 = as_type_ptr<op::Constant>(f->get_results().at(1)->get_argument(0));
    auto res3 = as_type_ptr<op::Constant>(f->get_results().at(2)->get_argument(0));
    auto res4 = as_type_ptr<op::Constant>(f->get_results().at(3)->get_argument(0));
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);
    ASSERT_TRUE(res4);

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51}),
              res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55}),
              res2_values);
    auto res3_values = res3->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59}),
              res3_values);
    auto res4_values = res4->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63}),
              res4_values);
}

TEST(constant_folding, constant_v1_split_axis_1_2_splits)
{
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = op::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = op::Constant::create(element::i64, Shape{}, {1});
    const auto num_splits = 2;

    auto split_v1 = make_shared<op::v1::Split>(const_data, const_axis, num_splits);
    auto f = make_shared<Function>(split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::Split>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), num_splits);

    auto res1 = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    auto res2 = as_type_ptr<op::Constant>(f->get_results().at(1)->get_argument(0));
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
                               32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55}),
              res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({8,  9,  10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31,
                               40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63}),
              res2_values);
}

TEST(constant_folding, constant_v1_variadic_split_axis_1_2_splits)
{
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = op::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = op::Constant::create(element::i16, Shape{}, {1});
    vector<int64_t> values_lengths{3, 1};
    auto constant_lengths =
        make_shared<op::Constant>(element::i64, Shape{values_lengths.size()}, values_lengths);

    auto variadic_split_v1 =
        make_shared<op::v1::VariadicSplit>(const_data, const_axis, constant_lengths);
    auto f = make_shared<Function>(variadic_split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::VariadicSplit>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), values_lengths.size());

    auto res1 = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    auto res2 = as_type_ptr<op::Constant>(f->get_results().at(1)->get_argument(0));
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}),
              res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47, 60, 61, 62, 63}),
              res2_values);
}

TEST(constant_folding, constant_v1_variadic_split_axis_1_3_splits_neg_length)
{
    vector<int64_t> data{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                         32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const auto const_data = op::Constant::create(element::i64, Shape{4, 4, 4}, data);
    const auto const_axis = op::Constant::create(element::i32, Shape{}, {1});
    vector<int64_t> values_lengths{1, 1, -1};
    auto constant_lengths =
        make_shared<op::Constant>(element::i64, Shape{values_lengths.size()}, values_lengths);

    auto variadic_split_v1 =
        make_shared<op::v1::VariadicSplit>(const_data, const_axis, constant_lengths);
    auto f = make_shared<Function>(variadic_split_v1->outputs(), ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::VariadicSplit>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), values_lengths.size());

    auto res1 = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    auto res2 = as_type_ptr<op::Constant>(f->get_results().at(1)->get_argument(0));
    auto res3 = as_type_ptr<op::Constant>(f->get_results().at(2)->get_argument(0));
    ASSERT_TRUE(res1);
    ASSERT_TRUE(res2);
    ASSERT_TRUE(res3);

    auto res1_values = res1->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51}),
              res1_values);
    auto res2_values = res2->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({4, 5, 6, 7, 20, 21, 22, 23, 36, 37, 38, 39, 52, 53, 54, 55}),
              res2_values);
    auto res3_values = res3->get_vector<int64_t>();
    ASSERT_EQ(vector<int64_t>({8,  9,  10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31,
                               40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63}),
              res3_values);
}

TEST(constant_folding, constant_v1_one_hot)
{
    vector<int64_t> indices{0, 1, 2};
    float16 on_value = 1.123f;
    float16 off_value = 0.321f;

    const auto indices_const = op::Constant::create(element::i64, Shape{3}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = op::Constant::create(element::f16, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::f16, Shape{}, {off_value});
    int64_t axis = 1;

    auto one_hot_v1 =
        make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    auto f = make_shared<Function>(one_hot_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::OneHot>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto res = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(res);

    ASSERT_EQ((Shape{3, 3}), res->get_output_shape(0));
    ASSERT_EQ(vector<float16>({on_value,
                               off_value,
                               off_value,
                               off_value,
                               on_value,
                               off_value,
                               off_value,
                               off_value,
                               on_value}),
              res->get_vector<float16>());
}

TEST(constant_folding, constant_v1_one_hot_negative_axes)
{
    vector<int64_t> indices{0, 2, -1, 1};
    int16_t on_value = 4;
    int16_t off_value = 1;

    const auto indices_const = op::Constant::create(element::i64, Shape{4}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = op::Constant::create(element::i16, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::i16, Shape{}, {off_value});
    int64_t axis = -1;

    auto one_hot_v1 =
        make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    auto f = make_shared<Function>(one_hot_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::OneHot>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto res = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(res);

    ASSERT_EQ((Shape{4, 3}), res->get_output_shape(0));
    ASSERT_EQ(vector<int16_t>({on_value,
                               off_value,
                               off_value,
                               off_value,
                               off_value,
                               on_value,
                               off_value,
                               off_value,
                               off_value,
                               off_value,
                               on_value,
                               off_value}),
              res->get_vector<int16_t>());
}

TEST(constant_folding, constant_v1_one_hot_negative_axes_2)
{
    vector<int64_t> indices{0, 2, 1, -1};
    auto on_value = true;
    auto off_value = false;

    const auto indices_const = op::Constant::create(element::i64, Shape{2, 2}, indices);
    const auto depth_const = op::Constant::create(element::i64, Shape{}, {3});
    const auto on_const = op::Constant::create(element::boolean, Shape{}, {on_value});
    const auto off_const = op::Constant::create(element::boolean, Shape{}, {off_value});
    int64_t axis = -1;

    auto one_hot_v1 =
        make_shared<op::v1::OneHot>(indices_const, depth_const, on_const, off_const, axis);
    auto f = make_shared<Function>(one_hot_v1, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v1::OneHot>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto res = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(res);

    ASSERT_EQ((Shape{2, 2, 3}), res->get_output_shape(0));
    ASSERT_EQ(vector<bool>({on_value,
                            off_value,
                            off_value,
                            off_value,
                            off_value,
                            on_value,
                            off_value,
                            on_value,
                            off_value,
                            off_value,
                            off_value,
                            off_value}),
              res->get_vector<bool>());
}

TEST(constant_folding, constant_tile_1d)
{
    Shape shape_in{2};
    Shape shape_repeats{1};
    Shape shape_out{4};

    vector<int> values_in{0, 1};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{2};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_3d_small_data_rank)
{
    Shape shape_in{2};
    Shape shape_repeats{3};
    Shape shape_out{2, 2, 4};

    vector<int> values_in{0, 1};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{2, 2, 2};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_3d_few_repeats)
{
    Shape shape_in{2, 1, 3};
    Shape shape_repeats{2};
    Shape shape_out{2, 2, 3};

    vector<int> values_in{1, 2, 3, 4, 5, 6};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{2, 1};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_1d_0_repeats)
{
    Shape shape_in{2};
    Shape shape_repeats{1};
    Shape shape_out{};

    vector<int> values_in{0, 1};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{0};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, constant_tile_0_rank_data)
{
    Shape shape_in{};
    Shape shape_repeats{1};
    Shape shape_out{4};

    vector<int> values_in{1};
    auto data = make_shared<op::Constant>(element::i32, shape_in, values_in);
    vector<int> values_repeats{4};
    auto repeats = make_shared<op::Constant>(element::i64, shape_repeats, values_repeats);
    auto tile = make_shared<op::v0::Tile>(data, repeats);
    auto f = make_shared<Function>(tile, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Tile>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> values_expected{1, 1, 1, 1};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::ConstantFolding>();
    ASSERT_FALSE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
    ASSERT_TRUE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}
