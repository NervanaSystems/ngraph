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

#include "ngraph/pass/constant_folding.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> values_permute{0, 0, 0, 0, 1, 1, 1, 1};
    ASSERT_EQ(values_permute, values_out);
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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int>();

    vector<int> padded_values{111, 777, 888, 111, 111};
    ASSERT_EQ(padded_values, values_out);
}

template <typename T>
static std::vector<T> get_result_constant(std::shared_ptr<Function> f, size_t pos)
{
    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(pos)->get_argument(0));
    return new_const->get_vector<T>();
}

TEST(constant_folding, constant_unary_binary)
{
    Shape shape_in{4};
    vector<int> values_a{1, 2, 3, 4};
    vector<int> values_b{1, 2, 3, 4};
    vector<int> values_c{-1, -1, -1, -1};
    vector<int> values_d{1, 4, 9, 16};
    auto a = make_shared<op::Constant>(element::i32, shape_in, values_a);
    auto b = make_shared<op::Constant>(element::i32, shape_in, values_b);
    auto c = make_shared<op::Constant>(element::i32, shape_in, values_c);
    auto d = make_shared<op::Constant>(element::i32, shape_in, values_d);

    auto add = a + b;
    auto sub = a - b;
    auto mul = a * b;
    auto divn = a / b;
    auto min = make_shared<op::Minimum>(c, a);
    auto max = make_shared<op::Maximum>(a, c);
    auto absn = make_shared<op::Abs>(c);
    auto neg = make_shared<op::Negative>(c);
    auto sqrt = make_shared<op::Sqrt>(d);
    auto neg_sqrt = make_shared<op::Sqrt>(c);

    auto f = make_shared<Function>(NodeVector{add, sub, mul, divn, min, max, absn, neg, sqrt},
                                   ParameterVector{});
    auto f_error = make_shared<Function>(NodeVector{neg_sqrt}, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    //expected values
    vector<int> add_expected{2, 4, 6, 8};
    vector<int> sub_expected{0, 0, 0, 0};
    vector<int> mul_expected{1, 4, 9, 16};
    vector<int> div_expected{1, 1, 1, 1};
    vector<int> min_expected{-1, -1, -1, -1};
    vector<int> max_expected{1, 2, 3, 4};
    vector<int> abs_neg_expected{1, 1, 1, 1};
    vector<int> sqrt_expected{1, 2, 3, 4};

    ASSERT_EQ(get_result_constant<int>(f, 0), add_expected);
    ASSERT_EQ(get_result_constant<int>(f, 1), sub_expected);
    ASSERT_EQ(get_result_constant<int>(f, 2), mul_expected);
    ASSERT_EQ(get_result_constant<int>(f, 3), div_expected);
    ASSERT_EQ(get_result_constant<int>(f, 4), min_expected);
    ASSERT_EQ(get_result_constant<int>(f, 5), max_expected);
    ASSERT_EQ(get_result_constant<int>(f, 6), abs_neg_expected);
    ASSERT_EQ(get_result_constant<int>(f, 7), abs_neg_expected);
    ASSERT_EQ(get_result_constant<int>(f, 8), sqrt_expected);
    ASSERT_ANY_THROW(pass_manager.run_passes(f_error));
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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
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

    auto result_as_shape_of =
        std::dynamic_pointer_cast<op::ShapeOf>(f->get_results().at(0)->get_argument(0));
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

    auto result_as_shape_of =
        std::dynamic_pointer_cast<op::ShapeOf>(f->get_results().at(0)->get_argument(0));
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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
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

    auto new_const =
        std::dynamic_pointer_cast<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<int32_t>();

    vector<int32_t> values_expected{6, 120, 504};
    ASSERT_EQ(values_expected, values_out);
}

TEST(constant_folding, pass_property)
{
    auto pass = std::make_shared<ngraph::pass::ConstantFolding>();
    ASSERT_EQ(false, pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
    ASSERT_EQ(false, pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
}
