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
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/util/attr_types.hpp"

#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, any_deduce)
{
    auto param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});

    auto r0 = make_shared<op::Any>(param_0, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::boolean);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::Any>(param_0, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::boolean);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::Any>(param_0, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::boolean);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::Any>(param_0, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::boolean);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, any_deduce_et_dynamic)
{
    auto param_0 = make_shared<op::Parameter>(element::dynamic, Shape{2, 4});

    auto r0 = make_shared<op::Any>(param_0, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::boolean);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::Any>(param_0, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::boolean);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::Any>(param_0, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::boolean);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::Any>(param_0, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::boolean);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, any_et_non_boolean)
{
    auto param_0 = make_shared<op::Parameter>(element::i32, Shape{2, 4});

    try
    {
        auto r = make_shared<op::Any>(param_0, AxisSet{0, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect invalid element type for Any";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element type must be boolean"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, any_axis_oob)
{
    auto param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});

    try
    {
        auto r = make_shared<op::Any>(param_0, AxisSet{0, 2, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for Any";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis (2) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, any_partial_rank_dynamic)
{
    auto param = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto axes = AxisSet{2385, 0, 4404}; // arbitrary
    auto any = make_shared<op::Any>(param, axes);

    EXPECT_EQ(any->get_output_element_type(0), element::boolean);
    EXPECT_TRUE(any->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, any_partial_rank_static_dynamic_ok_result_static)
{
    auto param = make_shared<op::Parameter>(element::boolean,
                                            PartialShape{1, 2, Dimension::dynamic(), 4, 5});
    auto axes = AxisSet{2, 3};
    auto any = make_shared<op::Any>(param, axes);

    EXPECT_EQ(any->get_output_element_type(0), element::boolean);
    EXPECT_EQ(any->get_shape(), (Shape{1, 2, 5}));
}

TEST(type_prop, any_partial_rank_static_dynamic_ok_result_dynamic)
{
    auto param = make_shared<op::Parameter>(
        element::boolean, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto axes = AxisSet{2, 3};
    auto any = make_shared<op::Any>(param, axes);

    EXPECT_EQ(any->get_output_element_type(0), element::boolean);
    EXPECT_TRUE(
        any->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(type_prop, any_partial_rank_static_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(
        element::boolean, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto axes = AxisSet{2, 5, 1};

    try
    {
        auto any = make_shared<op::Any>(param, axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for Any (rank-static dynamic input)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis (5) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, all_deduce)
{
    auto param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});

    auto r0 = make_shared<op::All>(param_0, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::boolean);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::All>(param_0, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::boolean);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::All>(param_0, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::boolean);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::All>(param_0, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::boolean);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, all_deduce_et_dynamic)
{
    auto param_0 = make_shared<op::Parameter>(element::dynamic, Shape{2, 4});

    auto r0 = make_shared<op::All>(param_0, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::boolean);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::All>(param_0, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::boolean);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::All>(param_0, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::boolean);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::All>(param_0, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::boolean);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, all_et_non_boolean)
{
    auto param_0 = make_shared<op::Parameter>(element::i32, Shape{2, 4});

    try
    {
        auto r = make_shared<op::All>(param_0, AxisSet{0, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect invalid element type for All";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element type must be boolean"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, all_axis_oob)
{
    auto param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});

    try
    {
        auto r = make_shared<op::All>(param_0, AxisSet{0, 2, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for All";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis (2) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, all_partial_rank_dynamic)
{
    auto param = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto axes = AxisSet{2385, 0, 4404}; // arbitrary
    auto all = make_shared<op::All>(param, axes);

    EXPECT_EQ(all->get_output_element_type(0), element::boolean);
    EXPECT_TRUE(all->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, all_partial_rank_static_dynamic_ok_result_static)
{
    auto param = make_shared<op::Parameter>(element::boolean,
                                            PartialShape{1, 2, Dimension::dynamic(), 4, 5});
    auto axes = AxisSet{2, 3};
    auto all = make_shared<op::All>(param, axes);

    EXPECT_EQ(all->get_output_element_type(0), element::boolean);
    EXPECT_EQ(all->get_shape(), (Shape{1, 2, 5}));
}

TEST(type_prop, all_partial_rank_static_dynamic_ok_result_dynamic)
{
    auto param = make_shared<op::Parameter>(
        element::boolean, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto axes = AxisSet{2, 3};
    auto all = make_shared<op::All>(param, axes);

    EXPECT_EQ(all->get_output_element_type(0), element::boolean);
    EXPECT_TRUE(
        all->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(type_prop, all_partial_rank_static_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(
        element::boolean, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto axes = AxisSet{2, 5, 1};

    try
    {
        auto all = make_shared<op::All>(param, axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for All (rank-static dynamic input)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis (5) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, DISABLED_benchmark_type_prop_add)
{
    auto p1 = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto p2 = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});

    constexpr size_t num_iterations = 1000000;
    size_t total_nanosec = 0;

    stopwatch sw;

    for (size_t i = 0; i < num_iterations; i++)
    {
        sw.start();
        auto n = make_shared<op::Add>(p1, p2);
        sw.stop();

        total_nanosec += sw.get_nanoseconds();
    }

    std::cout.imbue(std::locale(""));
    std::cout << "Constructed " << std::fixed << num_iterations << " Add ops in " << std::fixed
              << total_nanosec << " ns" << std::endl;
}

TEST(type_prop, DISABLED_benchmark_type_prop_convolution)
{
    auto d = make_shared<op::Parameter>(element::f32, Shape{64, 3, 224, 224});
    auto f = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 7});
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};

    constexpr size_t num_iterations = 1000000;
    size_t total_nanosec = 0;

    stopwatch sw;

    for (size_t i = 0; i < num_iterations; i++)
    {
        sw.start();
        auto n =
            make_shared<op::Convolution>(d, f, strides, dilation, padding_below, padding_above);
        sw.stop();

        total_nanosec += sw.get_nanoseconds();
    }

    std::cout.imbue(std::locale(""));
    std::cout << "Constructed " << std::fixed << num_iterations << " Convolution ops in "
              << std::fixed << total_nanosec << " ns" << std::endl;
}

TEST(type_prop, transpose_arg_static_input_order_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_static_input_order_constant_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = op::Constant::create(element::i64, Shape{4}, vector<int64_t>{2, 1, 0, 3});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{6, 4, 2, 8}));
}

TEST(type_prop, transpose_arg_static_input_order_constant_invalid_perm)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = op::Constant::create(element::i64, Shape{4}, vector<int64_t>{2, 9, 0, 3});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect invalid permutation";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Permutation AxisVector{2, 9, 0, 3} is not valid for input shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_static_input_order_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_static_input_order_static_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_static_input_order_rank_static_dynamic_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto input_order =
        make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_static_input_order_static_input_order_wrong_size)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{5});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order wrong size";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input order must have shape [n], where n is the rank of arg."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_static_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     transpose_arg_rank_static_dynamic_input_order_rank_static_dynamic_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order =
        make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_rank_static_dynamic_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order =
        make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_input_order_et_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::dynamic, Shape{4});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_input_order_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::boolean, Shape{4});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input element type not i64";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must have element type i64."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dyn_pad_pad_value_test)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3});
    auto pad_a = make_shared<op::Parameter>(element::i64, Shape{3});

    // padding value matches tensor data-type
    try
    {
        auto pad_v = make_shared<op::Parameter>(element::i32, Shape{});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Padding value and arg type mismatch");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    // padding value is scalar
    try
    {
        auto pad_v = make_shared<op::Parameter>(element::f32, Shape{3});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "DynPad arg is not scalar");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dyn_pad_wrong_ranks)
{
    auto pad_v = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3, 4});
        auto pad_a = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Shape of padding below must be of rank 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3});
        auto pad_a = make_shared<op::Parameter>(
            element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Shape of padding above must be of rank 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3});
        auto pad_a = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Arg and padding below ranks mismatch");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto pad_b = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
        auto pad_a = make_shared<op::Parameter>(element::i64, Shape{3});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Arg and padding above ranks mismatch");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
        auto pad_b = make_shared<op::Parameter>(element::i64, Shape{4});
        auto pad_a = make_shared<op::Parameter>(element::i64, Shape{3});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Padding below and above ranks mismatch");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dyn_pad_output_ranks_arg_static_ok)
{
    auto pad_v = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto pad_b = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto pad_a = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);

    EXPECT_EQ(dyn_pad->get_output_element_type(0), element::f32);
    EXPECT_TRUE(dyn_pad->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dyn_pad_output_ranks_arg_dynamic_ok)
{
    auto pad_v = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto pad_b = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto pad_a = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);

    EXPECT_EQ(dyn_pad->get_output_element_type(0), element::f32);
    EXPECT_TRUE(dyn_pad->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dyn_pad_output_ranks_pad_static_ok)
{
    auto pad_v = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3});
    auto pad_a = make_shared<op::Parameter>(element::i64, Shape{3});
    auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);

    EXPECT_EQ(dyn_pad->get_output_element_type(0), element::f32);
    EXPECT_TRUE(dyn_pad->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(3)));
}

TEST(type_prop, dynreshape_arg_static_pattern_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynreshape_arg_static_pattern_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_dynamic_pattern_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_dynamic_pattern_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::Transpose>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_zero)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 0, 2, 8});
    auto dynamic_arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = op::Constant::create(element::i64, Shape{4}, {1, 2, 0, 32});

    auto r1 = make_shared<op::DynReshape>(arg, pattern);
    EXPECT_EQ(r1->get_output_shape(0), (Shape{1, 2, 0, 32}));

    auto r2 = make_shared<op::DynReshape>(arg, pattern, true /*zero_flag*/);
    EXPECT_EQ(r2->get_output_shape(0), (Shape{1, 2, 2, 32}));

    auto r3 = make_shared<op::DynReshape>(dynamic_arg, pattern, true /*zero_flag*/);
    EXPECT_TRUE(
        r3->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic(), 32}));
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_negative)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 8});
    auto dynamic_arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = op::Constant::create(element::i64, Shape{4}, {1, 2, 4, -1});

    auto r1 = make_shared<op::DynReshape>(arg, pattern);
    EXPECT_EQ(r1->get_output_shape(0), (Shape{1, 2, 4, 16}));

    auto r2 = make_shared<op::DynReshape>(dynamic_arg, pattern);
    EXPECT_TRUE(
        r2->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, 4, Dimension::dynamic()}));
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_zero_negative)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 2, 0});
    auto dynamic_arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = op::Constant::create(element::i64, Shape{2}, {0, -1});

    auto r1 = make_shared<op::DynReshape>(arg, pattern);
    auto r2 = make_shared<op::DynReshape>(arg, pattern, true);
    EXPECT_EQ(r1->get_output_shape(0), (Shape{0, 0}));
    EXPECT_EQ(r2->get_output_shape(0), (Shape{2, 0}));

    auto r3 = make_shared<op::DynReshape>(dynamic_arg, pattern);
    auto r4 = make_shared<op::DynReshape>(dynamic_arg, pattern, true);
    EXPECT_TRUE(r3->get_output_partial_shape(0).same_scheme(PartialShape{0, Dimension::dynamic()}));
    EXPECT_TRUE(r4->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_negative_failure1)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 8});
    auto pattern = op::Constant::create(element::i64, Shape{4}, {1, 2, -1, -1});

    try
    {
        auto r = make_shared<op::DynReshape>(arg, pattern);
        FAIL() << "Expected failure on dynreshape construction";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("More than one dimension has size of -1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynreshape_arg_rank_static_pattern_negative_failure2)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 8});
    auto pattern = op::Constant::create(element::i64, Shape{4}, {1, 2, 4, -2});

    try
    {
        auto r = make_shared<op::DynReshape>(arg, pattern);
        FAIL() << "Expected failure on dynreshape construction";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dim size cannot be less than -1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

void DynReshape_Test_Shape_Except(const shared_ptr<Node>& param_0, const shared_ptr<Node>& param_1)
{
    try
    {
        auto r = make_shared<op::DynReshape>(param_0, param_1);
        FAIL() << "Did not detect parameter shape not rank 1";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("shape must have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynreshape_arg_static_pattern_static_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_arg_static_pattern_rank_static_dynamic_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_static_not_vector)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_arg_rank_static_dynamic_pattern_rank_static_dynamic_not_vector)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_arg_rank_dynamic_pattern_rank_static_dynamic_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pattern = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    DynReshape_Test_Shape_Except(arg, pattern);
}

TEST(type_prop, dynreshape_pattern_et_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::dynamic, Shape{4});

    auto r = make_shared<op::DynReshape>(arg, pattern);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynreshape_pattern_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = make_shared<op::Parameter>(element::boolean, Shape{4});

    try
    {
        auto r = make_shared<op::DynReshape>(arg, pattern);
        FAIL() << "Did not detect pattern elment type not i64";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Pattern must have element type i64."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynslice_arg_static_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynslice_arg_rank_static_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynslice_arg_static_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynslice_arg_rank_static_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynslice_arg_rank_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynslice_arg_rank_dynamic_params_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynslice_arg_rank_static_dynamic_params_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dynslice_static_shape)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5, 6});
    auto lower_bounds = op::Constant::create(element::i64, Shape{5}, {0, 1, 2, 3, 1});
    auto upper_bounds = op::Constant::create(element::i64, Shape{5}, {1, 3, 3, 5, 6});
    auto strides = op::Constant::create(element::i64, Shape{5}, {1, 1, 1, 2, 2});

    auto r = make_shared<op::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_shape(), (Shape{1, 2, 1, 1, 3}));
}

struct DynSliceParams
{
    std::vector<Shape> shapes;
    std::vector<std::vector<int64_t>> vals;
    std::vector<AxisSet> attrs;

    DynSliceParams(const std::vector<Shape>& shape,
                   const std::vector<std::vector<int64_t>>& val,
                   const std::vector<AxisSet>& attr)
        : shapes(shape)
        , vals(val)
        , attrs(attr)
    {
    }
};

struct DeduceDynSliceTest : ::testing::TestWithParam<DynSliceParams>
{
};

TEST_P(DeduceDynSliceTest, output_shape)
{
    auto tp = GetParam();
    auto arg = make_shared<op::Parameter>(element::f32, tp.shapes[0]);
    auto lower_bounds = op::Constant::create(element::i64, tp.shapes[1], tp.vals[0]);
    auto upper_bounds = op::Constant::create(element::i64, tp.shapes[2], tp.vals[1]);
    auto strides = op::Constant::create(element::i64, tp.shapes[3], tp.vals[2]);

    auto r = make_shared<op::DynSlice>(arg,
                                       lower_bounds,
                                       upper_bounds,
                                       strides,
                                       tp.attrs[0],
                                       tp.attrs[1],
                                       tp.attrs[2],
                                       tp.attrs[3],
                                       tp.attrs[4]);

    EXPECT_EQ(r->get_shape(), tp.shapes[4]);
}

INSTANTIATE_TEST_CASE_P(
    type_prop,
    DeduceDynSliceTest,
    ::testing::Values(
        // TODO(jbobba): These tests should pass.
        // DynSliceParams({{4}, {1}, {1}, {1}, {0}}, {{-9000}, {-8000}, {2}}, {{}, {}, {}, {}, {}}),
        // DynSliceParams({{5}, {1}, {1}, {1}, {0}}, {{3}, {2}, {1}}, {{}, {}, {}, {}, {}}),
        DynSliceParams({{2, 3, 4, 5, 6}, {5}, {5}, {5}, {1, 2, 1, 1, 3}},
                       {{0, 1, 2, 3, 1}, {1, 3, 3, 5, 6}, {1, 1, 1, 2, 2}},
                       {{}, {}, {}, {}, {}}),
        DynSliceParams({{10}, {0}, {0}, {0}, {10}}, {{}, {}, {}}, {{}, {}, {}, {}, {}}),
        DynSliceParams({{10}, {1}, {1}, {0}, {10}},
                       {{0}, {0}, {}},
                       {{}, {0}, {}, {}, {}}), // end-mask
        DynSliceParams({{10}, {1}, {1}, {0}, {9}},
                       {{-1}, {-1}, {}},
                       {{0}, {}, {}, {}, {}}), // begin-mask
        DynSliceParams({{10}, {1}, {1}, {0}, {10}}, {{0}, {10}, {}}, {{}, {}, {}, {}, {}}),
        DynSliceParams({{10}, {1}, {1}, {0}, {5}}, {{5}, {10}, {}}, {{}, {}, {}, {}, {}}),
        DynSliceParams({{10}, {1}, {1}, {0}, {5}}, {{-5}, {10}, {}}, {{}, {}, {}, {}, {}}),
        DynSliceParams({{10}, {1}, {1}, {1}, {6}},
                       {{-5}, {0}, {-1}}, // negative-stride
                       {{}, {0}, {}, {}, {}}),
        DynSliceParams({{10}, {1}, {1}, {1}, {3}}, {{-5}, {2}, {-1}}, {{}, {}, {}, {}, {}}),
        DynSliceParams({{10}, {1}, {1}, {1}, {5}}, {{0}, {0}, {2}}, {{}, {0}, {}, {}, {}}),
        DynSliceParams({{10}, {1}, {1}, {1}, {5}}, {{1}, {0}, {2}}, {{}, {0}, {}, {}, {}}),
        DynSliceParams({{10}, {1}, {1}, {1}, {10}}, {{-1}, {0}, {-1}}, {{}, {0}, {}, {}, {}}),
        DynSliceParams({{10}, {1}, {1}, {1}, {5}}, {{-1}, {0}, {-2}}, {{}, {0}, {}, {}, {}}),
        // Axis Masks: New, Shrink, Ellipsis
        DynSliceParams({{10}, {1}, {1}, {0}, {1, 10}}, {{0}, {10}, {}}, {{}, {}, {0}, {}, {}}),
        DynSliceParams({{1, 2, 3}, {2}, {2}, {0}, {1, 2, 2}},
                       {{0, 0}, {1, 2}, {}},
                       {{}, {}, {}, {}, {1}}),
        DynSliceParams({{1, 2, 3}, {4}, {4}, {0}, {1, 2, 1}},
                       {{0, 0, 0, 1}, {2, 3, 2, 2}, {}},
                       {{}, {}, {2}, {3}, {}}),
        DynSliceParams({{1, 2, 3}, {3}, {3}, {0}, {1, 1, 2, 1}},
                       {{0, 0, 1}, {2, 2, 2}, {}},
                       {{}, {}, {0}, {}, {1}}),
        DynSliceParams({{1, 2, 2, 2}, {1}, {1}, {1}, {1, 2, 2}},
                       {{-1}, {0}, {-2}},
                       {{1}, {1}, {}, {1}, {}}),
        DynSliceParams({{1, 2, 2, 2}, {4}, {4}, {0}, {1, 2, 2}},
                       {{0, 1, 0, 0}, {1, 2, 2, 2}, {}},
                       {{1}, {1}, {}, {1}, {}}),
        DynSliceParams({{1, 2, 3}, {3}, {3}, {0}, {1, 1, 2}},
                       {{0, 0, 1}, {2, 2, 2}, {}},
                       {{}, {}, {0}, {2}, {1}})));

void DynSlice_Test_Shape_Except(const shared_ptr<Node>& param_0,
                                const shared_ptr<Node>& param_1,
                                const shared_ptr<Node>& param_2,
                                const shared_ptr<Node>& param_3)
{
    try
    {
        auto r = make_shared<op::DynSlice>(param_0, param_1, param_2, param_3);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("shape must have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynslice_arg_static_params_rank_static_dynamic_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    {
        lower_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynSlice_Test_Shape_Except(arg, lower_bounds, upper_bounds, strides);
    }
    {
        lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});
        DynSlice_Test_Shape_Except(arg, lower_bounds, upper_bounds, strides);
    }
    {
        arg = make_shared<op::Parameter>(
            element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
        lower_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynSlice_Test_Shape_Except(arg, lower_bounds, upper_bounds, strides);
    }

    {
        upper_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynSlice_Test_Shape_Except(arg, lower_bounds, upper_bounds, strides);
    }
    {
        upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});
        DynSlice_Test_Shape_Except(arg, lower_bounds, upper_bounds, strides);
    }
    {
        arg = make_shared<op::Parameter>(
            element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
        upper_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynSlice_Test_Shape_Except(arg, lower_bounds, upper_bounds, strides);
    }

    {
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynSlice_Test_Shape_Except(arg, lower_bounds, upper_bounds, strides);
    }
    {
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});
        DynSlice_Test_Shape_Except(arg, lower_bounds, upper_bounds, strides);
    }
    {
        arg = make_shared<op::Parameter>(
            element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynSlice_Test_Shape_Except(arg, lower_bounds, upper_bounds, strides);
    }
}

TEST(type_prop, dynslice_params_et_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::DynSlice>(arg, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

void DynSlice_Test_Type_Except(const shared_ptr<Node>& param_0,
                               const shared_ptr<Node>& param_1,
                               const shared_ptr<Node>& param_2,
                               const shared_ptr<Node>& param_3)
{
    try
    {
        auto r = make_shared<op::DynSlice>(param_0, param_1, param_2, param_3);
        FAIL() << "Did not detect parameter element type not i64";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("must have element type i64."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynslice_params_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});

    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    {
        lower_bounds = make_shared<op::Parameter>(element::boolean, Shape{4});
        DynSlice_Test_Type_Except(arg, lower_bounds, upper_bounds, strides);
    }
    {
        upper_bounds = make_shared<op::Parameter>(element::boolean, Shape{4});
        DynSlice_Test_Type_Except(arg, lower_bounds, upper_bounds, strides);
    }
    {
        strides = make_shared<op::Parameter>(element::boolean, Shape{4});
        DynSlice_Test_Type_Except(arg, lower_bounds, upper_bounds, strides);
    }
}

TEST(type_prop, batchmatmul_deduce_3d)
{
    // Deduce type for matrix/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});
    auto bc = make_shared<op::BatchMatMul>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{5, 4, 3}));
}

TEST(type_prop, batchmatmul_deduce_left_rank_wrong)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 5, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{5, 2, 5});
    try
    {
        auto bc = make_shared<op::BatchMatMul>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("shape must have rank 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchmatmul_deduce_right_rank_wrong)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 5, 2, 5});
    try
    {
        auto bc = make_shared<op::BatchMatMul>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("shape must have rank 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchmatmul_deduce_element_type_mismatch)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{5, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::i32, Shape{5, 2, 5});
    try
    {
        auto bc = make_shared<op::BatchMatMul>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("compatible element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchmatmul_deduce_reduction_axes_size_mismatch)
{
    // Type deduction fails due to reduction axes size mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{6, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{6, 3, 5});
    try
    {
        auto bc = make_shared<op::BatchMatMul>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "BatchMatMul reduction axes size mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Product dimensions are not equal"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batchmatmul_partial_both_rank_dynamic_implicit)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMul>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().same_scheme(3));
}

TEST(type_prop, batchmatmul_partial_left_rank_dynamic_right_rank_static_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto d = make_shared<op::BatchMatMul>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().same_scheme(3));
}

TEST(type_prop, batchmatmul_partial_left_rank_static_dynamic_right_rank_dynamic)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMul>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().same_scheme(3));
}

TEST(type_prop, batchmatmul_partial_left_rank_static_dynamic_right_rank_static)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape{3, 4, 5});
    auto d = make_shared<op::BatchMatMul>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).same_scheme(PartialShape{3, 2, 5}));
}

TEST(type_prop, batchmatmul_partial_left_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMul>(param0, param1);

    ASSERT_EQ(d->get_output_element_type(0), element::f32);
}

TEST(type_prop, batchmatmul_partial_right_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMul>(param0, param1);

    ASSERT_EQ(d->get_output_element_type(0), element::i32);
}

TEST(type_prop, batchmatmul_partial_both_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto d = make_shared<op::BatchMatMul>(param0, param1);

    ASSERT_EQ(d->get_output_element_type(0), element::dynamic);
}

TEST(type_prop, prelu)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto slope = make_shared<op::Parameter>(element::f32, Shape{2});
    Shape prelu_shape{2, 4};
    auto prelu = make_shared<op::PRelu>(param, slope);
    ASSERT_EQ(prelu->get_element_type(), element::f32);
    ASSERT_EQ(prelu->get_shape(), prelu_shape);
}

TEST(type_prop, elu)
{
    Shape data_shape{2, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto alpha = make_shared<op::Parameter>(element::f32, Shape{});
    auto elu = make_shared<op::Elu>(data, alpha);
    ASSERT_EQ(elu->get_element_type(), element::f32);
    ASSERT_EQ(elu->get_shape(), data_shape);
}

TEST(type_prop, gather_no_axis)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather)
{
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I, 1);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, depth_to_space)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 128, 8, 8});
    auto space_to_depth = make_shared<op::DepthToSpace>(A, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 2, 64, 64}));
}

TEST(type_prop, space_to_depth)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, 8);

    ASSERT_EQ(space_to_depth->get_element_type(), element::f32);
    ASSERT_EQ(space_to_depth->get_shape(), (Shape{1, 128, 8, 8}));
}

TEST(type_prop, squeeze)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 4, 1, 8});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{0, 2});
    auto squeeze = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);
    ASSERT_EQ(squeeze->get_shape(), (Shape{4, 4, 1, 8}));

    axes_node = make_shared<ngraph::op::Constant>(element::u64, Shape{0}, vector<int64_t>{});
    auto squeeze_default_axes = make_shared<op::Squeeze>(param, axes_node);

    ASSERT_EQ(squeeze_default_axes->get_element_type(), element::f32);
    ASSERT_EQ(squeeze_default_axes->get_shape(), (Shape{4, 4, 8}));
}

TEST(type_prop, gather_nd_scalar_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_1d_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_scalar_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 3};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_1d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_2d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{1, 1};
    Shape out_shape{1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_scalar_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 2};
    Shape out_shape{2, 1};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_1d_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_scalar_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 3};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_1d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_2d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_fail_params_rank)
{
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    try
    {
        auto G = make_shared<op::Gather>(P, I, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect params rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("params rank is expected to be at least axis + 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_fail_indices_element_type)
{
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    try
    {
        auto G = make_shared<op::Gather>(P, I, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices element type must be i64 or i32"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_fail_params_rank)
{
    Shape params_shape{};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    try
    {
        auto G = make_shared<op::GatherND>(P, I);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect params rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("params rank is expected to be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_fail_indices_rank)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    try
    {
        auto G = make_shared<op::GatherND>(P, I);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("indices rank is expected to be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_fail_indices_element_type)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    try
    {
        auto G = make_shared<op::GatherND>(P, I);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices element type must be i64 or i32"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_add_fail_indices_element_type)
{
    Shape ref_shape{2, 3, 3};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2, 3, 3};
    Shape out_shape{2, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices element type must be i64 or i32"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_add_fail_updates_element_type)
{
    Shape ref_shape{2, 3, 3};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2, 3, 3};
    Shape out_shape{2, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::i32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Updates element type must be the same as Inputs"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_add_fail_updates_rank)
{
    Shape ref_shape{2, 3, 3};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 3, 3};
    Shape out_shape{2, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Updates rank is expected to be indices rank + inputs rank - 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_add_fail_updates_shape)
{
    Shape ref_shape{2, 3, 3};
    Shape indices_shape{2, 2};
    Shape updates_shape{1, 2, 3, 3};
    Shape out_shape{2, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Updates shape must be indices_shape + inputs_shape[1:]"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_indices_element_type)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices element type must be i64 or i32"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_indices_rank)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Indices rank is expected to be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_indices_last_dim)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{2, 4};
    Shape updates_shape{2, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices innermost dim";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Last dimension of indices can be at most the rank of inputs"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_updates_element_type)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::i32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Updates element type must be the same as inputs"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_updates_rank)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Rank of updates must be rank of inputs + rank of indices "
                                         "- last dimension of indices - 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_updates_shape)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{2, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Updates shape must be indices_shape[:-1] + inputs_shape[indices.shape[-1]:]"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, conv_bias_2d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{128});
    auto conv = make_shared<op::ConvolutionBias>(param0, param1, param2);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));

    EXPECT_EQ(conv->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_window_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv->get_data_dilation_strides(), (Strides{1, 1}));

    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, conv_bias_add_2d_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{128});
    auto param3 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91, 131});
    auto conv = make_shared<op::ConvolutionBiasAdd>(param0,
                                                    param1,
                                                    param2,
                                                    param3,
                                                    Strides{1, 1},
                                                    Strides{1, 1},
                                                    CoordinateDiff{0, 0},
                                                    CoordinateDiff{0, 0},
                                                    Strides{1, 1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));
}

TEST(type_prop, conv_bias_bprop_2d_deduce)
{
    // Deduce type
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10, 20});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{128});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91, 131});
    auto conv = make_shared<op::ConvolutionBiasBackpropFiltersBias>(data,
                                                                    filters->get_shape(),
                                                                    bias->get_shape(),
                                                                    delta,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{0, 0},
                                                                    CoordinateDiff{0, 0},
                                                                    Strides{1, 1});
    EXPECT_EQ(conv->get_output_element_type(0), element::f32);
    EXPECT_EQ(conv->get_output_element_type(1), element::f32);
    EXPECT_EQ(conv->get_output_shape(0), filters->get_shape());
    EXPECT_EQ(conv->get_output_shape(1), bias->get_shape());
}

TEST(type_prop, hardsigmoid)
{
    Shape data_shape{3, 5};
    float alpha = 0.1;
    float beta = 1.2;
    auto P = make_shared<op::Parameter>(element::f32, data_shape);
    auto H = make_shared<op::HardSigmoid>(P, alpha, beta);
    ASSERT_EQ(H->get_element_type(), element::f32);
    ASSERT_EQ(H->get_shape(), data_shape);
}

TEST(type_prop, group_conv)
{
    // Deduce type
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 4, 100, 150});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{128, 2, 10, 20});
    auto conv = make_shared<op::GroupConvolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1},
                                                  2);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));
}

TEST(type_prop, group_conv_auto)
{
    // Deduce type
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 4, 100, 150});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{128, 2, 10, 20});
    auto conv = make_shared<op::GroupConvolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1},
                                                  2,
                                                  op::PadType::AUTO);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 100, 150}));
    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{4, 9}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{5, 10}));
}

TEST(type_prop, group_conv_invalid_groups)
{
    // Deduce type
    try
    {
        auto conv = make_shared<op::GroupConvolution>(
            make_shared<op::Parameter>(element::f32, Shape{64, 20, 100, 150}),
            make_shared<op::Parameter>(element::f32, Shape{30, 10, 10, 20}),
            Strides{1, 1},
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1},
            3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid group conv";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data channels not a multiple of group size"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
    try
    {
        auto conv = make_shared<op::GroupConvolution>(
            make_shared<op::Parameter>(element::f32, Shape{64, 30, 100, 150}),
            make_shared<op::Parameter>(element::f32, Shape{20, 10, 10, 20}),
            Strides{1, 1},
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1},
            3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid group conv";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("# Filters not a multiple of group size"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
    try
    {
        auto conv = make_shared<op::GroupConvolution>(
            make_shared<op::Parameter>(element::f32, Shape{64, 30, 100, 150}),
            make_shared<op::Parameter>(element::f32, Shape{30, 20, 10, 20}),
            Strides{1, 1},
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1},
            3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid group conv";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incorrect number of channels per filter"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, normalize_invalid_input_tensor_rank)
{
    Shape data_shape{1, 2, 3, 4, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto scale = make_shared<op::Parameter>(element::f32, Shape{});
    bool across_spatial{false};
    bool channel_shared{true};
    float eps{1e-6f};

    try
    {
        auto normalize =
            make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input tensor rank must be 2, 3 or 4 dimensional"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    data = make_shared<op::Parameter>(element::f32, Shape{2});

    try
    {
        auto normalize =
            make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input tensor rank must be 2, 3 or 4 dimensional"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, normalize_invalid_scale_rank)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto scale = make_shared<op::Parameter>(element::f32, Shape{3});
    bool across_spatial{false};
    bool channel_shared{true};
    float eps{1e-6f};

    try
    {
        auto normalize =
            make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Scale must be a scalar if 'channels_shared' "
                                         "parameter is true"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    channel_shared = false;
    try
    {
        auto normalize =
            make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Scale must be a vector of size of input tensor "
                                         "channels"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    data = make_shared<op::Parameter>(element::f32, Shape{4, 3});
    try
    {
        auto normalize =
            make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Scale must be a scalar if input tensor is of rank 2"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, normalize)
{
    Shape data_shape{2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto scale = make_shared<op::Parameter>(element::f32, Shape{2});
    bool across_spatial{false};
    bool channel_shared{false};
    float eps{1e-6f};

    auto normalize = make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
    EXPECT_EQ(normalize->get_element_type(), element::f32);
    EXPECT_EQ(normalize->get_shape(), (Shape{2, 3, 4}));
}

TEST(type_prop, function_revalidate_and_infer)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = op::Constant::create(element::i64, Shape{6}, {1, 3, 16, 2, 2, 2});

    auto r = make_shared<op::DynReshape>(arg, pattern);
    auto relu = make_shared<op::Relu>(r);
    auto f = make_shared<Function>(relu, ParameterVector{arg});

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_shape(0), (Shape{1, 3, 16, 2, 2, 2}));
    EXPECT_EQ(f->get_output_shape(0), (Shape{1, 3, 16, 2, 2, 2}));

    auto new_pattern = op::Constant::create(element::i64, Shape{2}, {32, 12});
    r->input(1).replace_source_output(new_pattern->output(0));

    f->validate_nodes_and_infer_types();
    EXPECT_EQ(r->get_output_shape(0), (Shape{32, 12}));
    EXPECT_EQ(f->get_output_shape(0), (Shape{32, 12}));
}

TEST(type_prop, gemm)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{6, 4});
    auto C = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto gemm_func = make_shared<op::Gemm>(A, B, C);
    EXPECT_EQ(gemm_func->get_element_type(), element::f32);
    EXPECT_EQ(gemm_func->get_shape(), (Shape{3, 4}));
}

TEST(type_prop, gemm_broadcast_input_C)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{6, 4});
    auto C = make_shared<op::Parameter>(element::f32, Shape{});
    auto gemm_func = make_shared<op::Gemm>(A, B, C);
    EXPECT_EQ(gemm_func->get_element_type(), element::f32);
    EXPECT_EQ(gemm_func->get_shape(), (Shape{3, 4}));
}

TEST(type_prop, grn)
{
    float bias = 1.25f;
    Shape data_shape{2, 3, 4, 5};
    auto A = make_shared<op::Parameter>(element::f32, data_shape);
    auto grn = make_shared<op::GRN>(A, bias);

    ASSERT_EQ(grn->get_element_type(), element::f32);
    ASSERT_EQ(grn->get_shape(), data_shape);
}

TEST(type_prop, grn_invalid_data_rank)
{
    float bias = 1.25f;
    auto A = make_shared<op::Parameter>(element::f32, Shape{4});

    try
    {
        auto grn = make_shared<op::GRN>(A, bias);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input tensor rank must be 2, 3 or 4 dimensional"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4, 5});

    try
    {
        auto grn = make_shared<op::GRN>(A, bias);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input tensor rank must be 2, 3 or 4 dimensional"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, mvn)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto mvn_func = make_shared<op::MVN>(data);
    EXPECT_EQ(mvn_func->get_element_type(), element::f32);
    EXPECT_EQ(mvn_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, fused_clamp)
{
    const auto data = make_shared<op::Parameter>(element::f64, Shape{2, 2});

    try
    {
        const auto clamp = make_shared<op::Clamp>(data, 2.0, 1.0);
        EXPECT_FALSE(clamp.get())
            << "Clamp validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("The 'min' parameter needs to be less than 'max' for Clamp"));
    }

    const auto clamp = make_shared<op::Clamp>(data, 1.0, 2.0);
    EXPECT_EQ(clamp->get_element_type(), element::f64);
    EXPECT_EQ(clamp->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, leaky_relu)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto alpha = make_shared<op::Parameter>(element::f32, Shape{});
    auto leaky_relu_func = make_shared<op::LeakyRelu>(data, alpha);
    EXPECT_EQ(leaky_relu_func->get_element_type(), element::f32);
    EXPECT_EQ(leaky_relu_func->get_shape(), (Shape{3, 6}));
}

TEST(type_prop, unsqueeze)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{1, 2});
    auto squeeze = make_shared<op::Unsqueeze>(param, axes_node);

    ASSERT_EQ(squeeze->get_element_type(), element::f32);
    ASSERT_EQ(squeeze->get_shape(), (Shape{4, 1, 1, 1, 4, 1, 8}));
}

TEST(type_prop, scale_shift_no_broadcast)
{
    auto data = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto scale = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto shift = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto scale_shift_func = make_shared<op::ScaleShift>(data, scale, shift);
    EXPECT_EQ(scale_shift_func->get_element_type(), element::f64);
    EXPECT_EQ(scale_shift_func->get_shape(), (Shape{3, 6}));
}

TEST(type_prop, scale_shift)
{
    auto data = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto scale = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto shift = make_shared<op::Parameter>(element::f64, Shape{});
    auto scale_shift_func = make_shared<op::ScaleShift>(data, scale, shift);
    EXPECT_EQ(scale_shift_func->get_element_type(), element::f64);
    EXPECT_EQ(scale_shift_func->get_shape(), (Shape{3, 6}));
}

TEST(type_prop, shuffle_channels_axis_validation)
{
    try
    {
        const auto data = make_shared<op::Parameter>(element::f64, Shape{1, 2, 3, 4});
        const auto shuffle_channels = make_shared<op::ShuffleChannels>(data, -5, 5);
        FAIL() << "ShuffleChannels validation did not work. Op node was created with incorrect "
                  "params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The 'axis' parameter for ShuffleChannels has to point to one of the "
                             "input tensor's shape dimensions");
    }
}

TEST(type_prop, shuffle_channels_negative_axis_calculation)
{
    const auto data = make_shared<op::Parameter>(element::f64, Shape{1, 2, 3, 4});

    const auto shuffle_channels = make_shared<op::ShuffleChannels>(data, -3, 2);

    EXPECT_EQ(shuffle_channels->get_zero_based_axis(), 1);
}

TEST(type_prop, shuffle_channels_invalid_input_shape)
{
    try
    {
        const auto data = make_shared<op::Parameter>(element::f64, Shape{});
        const auto shuffle_channels = make_shared<op::ShuffleChannels>(data, 0, 1);
        FAIL() << "ShuffleChannels validation did not work. Op node was created with incorrect "
                  "params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The input tensor's shape is expected to be at least 1D.");
    }
}

TEST(type_prop, shuffle_channels_invalid_groups_value)
{
    try
    {
        const auto data = make_shared<op::Parameter>(element::f64, Shape{1, 2, 3, 15});
        const auto shuffle_channels = make_shared<op::ShuffleChannels>(data, -1, 2);
        FAIL() << "ShuffleChannels validation did not work. Op node was created with incorrect "
                  "params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "The channel dimension size has to be a multiple of the groups parameter value.");
    }
}

TEST(type_prop, squared_difference)
{
    const auto x1 = make_shared<op::Parameter>(element::f64, Shape{2, 2});
    const auto x2 = make_shared<op::Parameter>(element::f64, Shape{3, 2});
    const auto x3 = make_shared<op::Parameter>(element::f64, Shape{1, 2});

    try
    {
        const auto squared_diff = make_shared<op::SquaredDifference>(x1, x2);
        FAIL() << "SquaredDifference node was created with incorrect data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("axes are incompatible"));
    }

    const auto clamp = make_shared<op::SquaredDifference>(x1, x3);
    EXPECT_EQ(clamp->get_element_type(), element::f64);
    EXPECT_EQ(clamp->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, split)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});

    try
    {
        const std::vector<size_t> splits = {1, 6}; // should sum up to 6
        const auto split = make_shared<op::Split>(data, 1, splits);
        FAIL() << "Split node was created with incorrect data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("has to be equal to the sum of splits passed to the op: 7"));
    }

    try
    {
        const std::vector<size_t> splits = {4, 2};
        const auto split = make_shared<op::Split>(data, -5, splits); //invalid axis
        FAIL() << "Split node was created with incorrect data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The 'axis' parameter for Split has to point to one of "
                                         "the input tensor's shape dimensions."));
    }

    const auto split = make_shared<op::Split>(data, 1, 2);
    EXPECT_EQ(split->outputs().size(), 2);
    EXPECT_EQ(split->output(0).get_shape(), (Shape{2, 3}));
    EXPECT_EQ(split->output(1).get_shape(), (Shape{2, 3}));
    EXPECT_EQ(split->output(0).get_element_type(), element::i32);
    EXPECT_EQ(split->output(1).get_element_type(), element::i32);
}

TEST(type_prop, lstm_cell)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(X, W, R, H_t, C_t, hidden_size);
    EXPECT_EQ(lstm_cell->output(0).get_element_type(), element::f32);
    EXPECT_EQ(lstm_cell->output(0).get_shape(), (Shape{batch_size, hidden_size}));
    EXPECT_EQ(lstm_cell->output(1).get_element_type(), element::f32);
    EXPECT_EQ(lstm_cell->output(1).get_shape(), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, lstm_cell_invalid_input)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<op::Parameter>(element::f32, Shape{1 * hidden_size, input_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, W, R, H_t, C_t, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor W must have shape"));
    }

    // Invalid R tensor shape.
    W = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    R = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, 1});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, W, R, H_t, C_t, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor R must have shape"));
    }

    // Invalid H_t tensor shape.
    R = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    H_t = make_shared<op::Parameter>(element::f32, Shape{4, hidden_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, W, R, H_t, C_t, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor H_t must have shape"));
    }

    // Invalid C_t tensor shape.
    H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    C_t = make_shared<op::Parameter>(element::f32, Shape{4, hidden_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, W, R, H_t, C_t, hidden_size);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor C_t must have shape"));
    }

    // Invalid B tensor shape.
    C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto B = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size});
    auto P = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, W, R, H_t, C_t, hidden_size, B, P);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor B must have shape"));
    }

    // Invalid P tensor shape.
    B = make_shared<op::Parameter>(element::f32, Shape{2 * gates_count * hidden_size});
    P = make_shared<op::Parameter>(element::f32, Shape{hidden_size});
    try
    {
        const auto lstm_cell = make_shared<op::LSTMCell>(X, W, R, H_t, C_t, hidden_size, B, P);
        FAIL() << "LSTMCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor P must have shape"));
    }
}

TEST(type_prop, fake_quantize)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});
    const int levels = 5;

    const auto fake_quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    EXPECT_EQ(fake_quantize->get_element_type(), element::f32);
    EXPECT_EQ(fake_quantize->get_shape(), (Shape{1, 2, 3, 4}));
}

TEST(type_prop, fake_quantize_invalid_rank)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto input_low = make_shared<op::Parameter>(element::f32, Shape{3});
    auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    auto output_high = make_shared<op::Parameter>(element::f32, Shape{});
    const int levels = 5;

    // Invalid input_low dimension
    try
    {
        const auto fake_quantize = make_shared<op::FakeQuantize>(
            data, input_low, input_high, output_low, output_high, levels);
        EXPECT_FALSE(fake_quantize.get())
            << "FakeQuantize validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("must either be a scalar or a vector of size equal "
                                         "to number of channels."));
    }

    // Invalid input_high dimension
    input_low = make_shared<op::Parameter>(element::f32, Shape{});
    input_high = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        const auto fake_quantize = make_shared<op::FakeQuantize>(
            data, input_low, input_high, output_low, output_high, levels);
        EXPECT_FALSE(fake_quantize.get())
            << "FakeQuantize validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("must either be a scalar or a vector of size equal "
                                         "to number of channels."));
    }

    // Invalid output_low dimension
    input_high = make_shared<op::Parameter>(element::f32, Shape{});
    output_low = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        const auto fake_quantize = make_shared<op::FakeQuantize>(
            data, input_low, input_high, output_low, output_high, levels);
        EXPECT_FALSE(fake_quantize.get())
            << "FakeQuantize validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("must either be a scalar or a vector of size equal "
                                         "to number of channels."));
    }

    // Invalid output_high dimension
    output_low = make_shared<op::Parameter>(element::f32, Shape{});
    output_high = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        const auto fake_quantize = make_shared<op::FakeQuantize>(
            data, input_low, input_high, output_low, output_high, levels);
        EXPECT_FALSE(fake_quantize.get())
            << "FakeQuantize validation did not work. Op node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("must either be a scalar or a vector of size equal "
                                         "to number of channels."));
    }
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_static_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_rank_static_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_rank_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_arg_rank_static_dynamic_replacement_static_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_static_dynamic_replacement_rank_static_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop, dynreplaceslice_arg_rank_static_dynamic_replacement_rank_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_static_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 4});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop,
     dynreplaceslice_arg_static_replacement_rank_static_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_rank_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_static_dynamic_replacement_static_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(
    type_prop,
    dynreplaceslice_arg_rank_static_dynamic_replacement_rank_static_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_static_dynamic_replacement_rank_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto upper_bounds =
        make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8}));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_static_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_rank_static_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_rank_dynamic_params_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape{4});
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_static_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_dynamic_replacement_rank_static_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_dynamic_replacement_rank_dynamic_params_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_static_params_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop,
     dynreplaceslice_arg_rank_dynamic_replacement_rank_static_dynamic_params_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    // TODO(amprocte): We should be able to infer PartialShape::dynamic(4) here.
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_arg_rank_dynamic_replacement_rank_dynamic_params_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, dynreplaceslice_static_shape)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5, 6});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{1, 2, 1, 1, 3});
    auto lower_bounds = op::Constant::create(element::i64, Shape{5}, {0, 1, 2, 3, 1});
    auto upper_bounds = op::Constant::create(element::i64, Shape{5}, {1, 3, 3, 5, 6});
    auto strides = op::Constant::create(element::i64, Shape{5}, {1, 1, 1, 2, 2});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_shape(), (Shape{2, 3, 4, 5, 6}));
}

TEST(type_prop, dynreplaceslice_static_shape_replacement_inconsistent)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5, 6});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 1, 1, 4});
    auto lower_bounds = op::Constant::create(element::i64, Shape{5}, {0, 1, 2, 3, 1});
    auto upper_bounds = op::Constant::create(element::i64, Shape{5}, {1, 3, 3, 5, 6});
    auto strides = op::Constant::create(element::i64, Shape{5}, {1, 1, 1, 2, 2});

    try
    {
        auto r =
            make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);
        FAIL() << "Did not detect mismatch of replacement shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), "Shape of the replacement is not compatible with the shape of the slice");
    }
}

struct DynReplaceSliceParams
{
    Shape arg_shape;
    Shape lower_bounds_shape;
    Shape upper_bounds_shape;
    Shape strides_shape;
    Shape replacement_shape;

    std::vector<int64_t> lower_bounds_val;
    std::vector<int64_t> upper_bounds_val;
    std::vector<int64_t> strides_val;

    AxisSet lower_bounds_mask;
    AxisSet upper_bounds_mask;
    AxisSet new_axis;
    AxisSet shrink_axis;
    AxisSet ellipsis_mask;
};

struct DeduceDynReplaceSliceTest : ::testing::TestWithParam<DynReplaceSliceParams>
{
};

TEST_P(DeduceDynReplaceSliceTest, output_shape)
{
    auto tp = GetParam();
    auto arg = make_shared<op::Parameter>(element::f32, tp.arg_shape);
    auto replacement = make_shared<op::Parameter>(element::f32, tp.replacement_shape);
    auto lower_bounds =
        op::Constant::create(element::i64, tp.lower_bounds_shape, tp.lower_bounds_val);
    auto upper_bounds =
        op::Constant::create(element::i64, tp.upper_bounds_shape, tp.upper_bounds_val);
    auto strides = op::Constant::create(element::i64, tp.strides_shape, tp.strides_val);

    auto r = make_shared<op::DynReplaceSlice>(arg,
                                              replacement,
                                              lower_bounds,
                                              upper_bounds,
                                              strides,
                                              tp.lower_bounds_mask,
                                              tp.upper_bounds_mask,
                                              tp.new_axis,
                                              tp.shrink_axis,
                                              tp.ellipsis_mask);

    EXPECT_EQ(r->get_shape(), tp.arg_shape);
}

INSTANTIATE_TEST_CASE_P(
    type_prop,
    DeduceDynReplaceSliceTest,
    ::testing::Values(
        DynReplaceSliceParams{{2, 3, 4, 5, 6},
                              {5},
                              {5},
                              {5},
                              {1, 2, 1, 1, 3},
                              {0, 1, 2, 3, 1},
                              {1, 3, 3, 5, 6},
                              {1, 1, 1, 2, 2},
                              {},
                              {},
                              {},
                              {},
                              {}},
        DynReplaceSliceParams{{10}, {0}, {0}, {0}, {10}, {}, {}, {}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{
            {10}, {1}, {1}, {0}, {10}, {0}, {0}, {}, {}, {0}, {}, {}, {}}, // end-mask
        DynReplaceSliceParams{
            {10}, {1}, {1}, {0}, {9}, {-1}, {-1}, {}, {0}, {}, {}, {}, {}}, // begin-mask
        DynReplaceSliceParams{{10}, {1}, {1}, {0}, {10}, {0}, {10}, {}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {0}, {5}, {5}, {10}, {}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {0}, {5}, {-5}, {10}, {}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{{10},
                              {1},
                              {1},
                              {1},
                              {6},
                              {-5},
                              {0},
                              {-1}, // negative-stride
                              {},
                              {0},
                              {},
                              {},
                              {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {3}, {-5}, {2}, {-1}, {}, {}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {5}, {0}, {0}, {2}, {}, {0}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {5}, {1}, {0}, {2}, {}, {0}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {10}, {-1}, {0}, {-1}, {}, {0}, {}, {}, {}},
        DynReplaceSliceParams{{10}, {1}, {1}, {1}, {5}, {-1}, {0}, {-2}, {}, {0}, {}, {}, {}},
        /* Axis Masks: New, Shrink, Ellipsis */
        DynReplaceSliceParams{{10}, {1}, {1}, {0}, {1, 10}, {0}, {10}, {}, {}, {}, {0}, {}, {}},
        DynReplaceSliceParams{
            {1, 2, 3}, {2}, {2}, {0}, {1, 2, 2}, {0, 0}, {1, 2}, {}, {}, {}, {}, {}, {1}},
        DynReplaceSliceParams{{1, 2, 3},
                              {4},
                              {4},
                              {0},
                              {1, 2, 1},
                              {0, 0, 0, 1},
                              {2, 3, 2, 2},
                              {},
                              {},
                              {},
                              {2},
                              {3},
                              {}},
        DynReplaceSliceParams{
            {1, 2, 3}, {3}, {3}, {0}, {1, 1, 2, 1}, {0, 0, 1}, {2, 2, 2}, {}, {}, {}, {0}, {}, {1}},
        DynReplaceSliceParams{
            {1, 2, 2, 2}, {1}, {1}, {1}, {1, 2, 2}, {-1}, {0}, {-2}, {1}, {1}, {}, {1}, {}},
        DynReplaceSliceParams{{1, 2, 2, 2},
                              {4},
                              {4},
                              {0},
                              {1, 2, 2},
                              {0, 1, 0, 0},
                              {1, 2, 2, 2},
                              {},
                              {1},
                              {1},
                              {},
                              {1},
                              {}},
        DynReplaceSliceParams{
            {1, 2, 3}, {3}, {3}, {0}, {1, 1, 2}, {0, 0, 1}, {2, 2, 2}, {}, {}, {}, {0}, {2}, {1}}));

void DynReplaceSlice_Test_Shape_Except(const shared_ptr<Node>& param_0,
                                       const shared_ptr<Node>& param_1,
                                       const shared_ptr<Node>& param_2,
                                       const shared_ptr<Node>& param_3,
                                       const shared_ptr<Node>& param_4)
{
    try
    {
        auto r = make_shared<op::DynReplaceSlice>(param_0, param_1, param_2, param_3, param_4);
        FAIL() << "Did not detect attributes not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("shape must have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynreplaceslice_arg_static_replacement_static_params_rank_static_dynamic_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto strides = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    {
        lower_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        lower_bounds = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        arg = make_shared<op::Parameter>(
            element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
        lower_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }

    {
        upper_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        upper_bounds = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        arg = make_shared<op::Parameter>(
            element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
        upper_bounds =
            make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }

    {
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        arg = make_shared<op::Parameter>(
            element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        replacement =
            make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 2, 4});
        strides = make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});
        DynReplaceSlice_Test_Shape_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
}

TEST(type_prop, dynreplaceslice_params_et_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::dynamic, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::dynamic, Shape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::dynamic, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::dynamic, Shape{4});
    auto strides = make_shared<op::Parameter>(element::dynamic, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::dynamic);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

TEST(type_prop, dynreplaceslice_params_et_dynamic_inferrable_ok)
{
    auto arg = make_shared<op::Parameter>(element::dynamic, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::boolean, Shape{2, 4, 2, 4});
    auto lower_bounds = make_shared<op::Parameter>(element::dynamic, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::dynamic, Shape{4});
    auto strides = make_shared<op::Parameter>(element::dynamic, Shape{4});

    auto r =
        make_shared<op::DynReplaceSlice>(arg, replacement, lower_bounds, upper_bounds, strides);

    EXPECT_EQ(r->get_output_element_type(0), element::boolean);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
}

void DynReplaceSlice_Test_Type_Except(const shared_ptr<Node>& param_0,
                                      const shared_ptr<Node>& param_1,
                                      const shared_ptr<Node>& param_2,
                                      const shared_ptr<Node>& param_3,
                                      const shared_ptr<Node>& param_4)
{
    try
    {
        auto r = make_shared<op::DynReplaceSlice>(param_0, param_1, param_2, param_3, param_4);
        FAIL() << "Did not detect parameter element type not i64";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("must have element type i64."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dynreplaceslice_params_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto replacement = make_shared<op::Parameter>(element::f32, Shape{2, 4, 2, 4});

    auto lower_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto upper_bounds = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    {
        lower_bounds = make_shared<op::Parameter>(element::boolean, Shape{4});
        DynReplaceSlice_Test_Type_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        upper_bounds = make_shared<op::Parameter>(element::boolean, Shape{4});
        DynReplaceSlice_Test_Type_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
    {
        strides = make_shared<op::Parameter>(element::boolean, Shape{4});
        DynReplaceSlice_Test_Type_Except(arg, replacement, lower_bounds, upper_bounds, strides);
    }
}

TEST(type_prop, group_conv_transpose)
{
    // C x M / group x kH x kW
    auto weights = make_shared<op::Parameter>(element::f32, Shape{16, 2, 3, 3});
    // N x C x H x W
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 6, 6});
    auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                          weights,
                                                          Strides{1, 1},
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          2);
    EXPECT_EQ(gct->get_element_type(), element::f32);
    EXPECT_EQ(gct->get_shape(), (Shape{1, 4, 8, 8}));
    EXPECT_EQ(gct->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(gct->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(gct->get_padding_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gct->get_padding_end(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gct->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gct->get_groups(), size_t(2));
    EXPECT_EQ(gct->get_pad_type(), op::PadType::EXPLICIT);
}

TEST(type_prop, group_conv_transpose_output_shape)
{
    // N x C x H x W
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
    // C x M / group x kH x kW
    auto weights = make_shared<op::Parameter>(element::f32, Shape{16, 2, 3, 3});
    auto gct = make_shared<op::GroupConvolutionTranspose>(
        data, weights, Strides{1, 1}, Strides{1, 1}, CoordinateDiff{0, 0}, Shape{1, 2, 3, 3}, 1);
    EXPECT_EQ(gct->get_element_type(), element::f32);
    EXPECT_EQ(gct->get_shape(), (Shape{1, 2, 3, 3}));
    EXPECT_EQ(gct->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(gct->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(gct->get_padding_begin(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(gct->get_padding_end(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(gct->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gct->get_groups(), size_t(1));
    EXPECT_EQ(gct->get_pad_type(), op::PadType::EXPLICIT);
}

TEST(type_prop, group_conv_transpose_invalid_params)
{
    // C x M / group x kH x kW
    auto weights = make_shared<op::Parameter>(element::f32, Shape{16, 20, 3, 3});
    // N x C x H x W
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    21);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incorrect value of groups:"));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    5);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of data channels not a multiple of group size."));
    }

    try
    {
        // C x M / group x kH x kW
        auto bad_weights = make_shared<op::Parameter>(element::f32, Shape{10, 20, 3, 3});
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    bad_weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    8);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of filters channels must be equal to number of ") +
                                 std::string("data channels"));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    4,
                                                                    op::PadType::SAME_UPPER);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Currently only eplicit pad type is supported."));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    4);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Strides should be of number of input data features size."));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0},
                                                                    4);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Dilations should be of number of input data features size."));
    }

    try
    {
        const auto gct = make_shared<op::GroupConvolutionTranspose>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{2, 2},
                                                                    CoordinateDiff{0, 0, 1, 1},
                                                                    4);
        EXPECT_FALSE(gct.get()) << "GroupConvolutionTranspose validation did not work. "
                                   "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Output padding should be of number of input data features size."));
    }
}

TEST(type_prop, range_nonconst_ok)
{
    auto start = make_shared<op::Parameter>(element::i32, Shape{});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::i32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_some_dyn_et_ok)
{
    auto start = make_shared<op::Parameter>(element::i32, Shape{});
    auto stop = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto step = make_shared<op::Parameter>(element::i32, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::i32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_all_dyn_et_ok)
{
    auto start = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto stop = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto step = make_shared<op::Parameter>(element::dynamic, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::dynamic);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_f32_ok)
{
    auto start = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Parameter>(element::dynamic, Shape{});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::f32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_nonconst_boolean_fails)
{
    auto start = make_shared<op::Parameter>(element::dynamic, Shape{});
    auto stop = make_shared<op::Parameter>(element::boolean, Shape{});
    auto step = make_shared<op::Parameter>(element::dynamic, Shape{});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "Boolean element type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Element type for start, stop, and step, must not be boolean.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_ok)
{
    auto start = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{3});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{2});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), element::i32);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, range_some_const_zero_stride_fails)
{
    auto start = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{3});
    auto stop = make_shared<op::Parameter>(element::i32, Shape{});
    auto step = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{0});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "Zero stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_plus_inf_start_fails)
{
    auto start = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{std::numeric_limits<float>::infinity()});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "+Infinity start not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_minus_inf_start_fails)
{
    auto start = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{-std::numeric_limits<float>::infinity()});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "-Infinity start not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_nan_start_fails)
{
    auto start =
        make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "NaN start not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'start' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_plus_inf_stop_fails)
{
    auto start = make_shared<op::Parameter>(element::f32, Shape{});
    auto stop = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{std::numeric_limits<float>::infinity()});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "+Infinity stop not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_minus_inf_stop_fails)
{
    auto start = make_shared<op::Parameter>(element::f32, Shape{});
    auto stop = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{-std::numeric_limits<float>::infinity()});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "-Infinity stop not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_nan_stio_fails)
{
    auto start = make_shared<op::Parameter>(element::f32, Shape{});
    auto stop = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{1});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "NaN stop not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'stop' cannot be nan or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_plus_inf_stride_fails)
{
    auto start = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{3});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{std::numeric_limits<float>::infinity()});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "+Infinity stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero, nan, or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_minus_inf_stride_fails)
{
    auto start = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{3});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{-std::numeric_limits<float>::infinity()});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "-Infinity stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero, nan, or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_some_const_nan_stride_fails)
{
    auto start = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{3});
    auto stop = make_shared<op::Parameter>(element::f32, Shape{});
    auto step = make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{std::nanf("")});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "NaN stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero, nan, or infinite.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, range_all_const_zero_stride_fails)
{
    auto start = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{3});
    auto stop = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{5});
    auto step = make_shared<op::Constant>(element::i32, Shape{}, std::vector<int32_t>{0});

    try
    {
        auto range = make_shared<op::Range>(start, stop, step);
        FAIL() << "Zero stride not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "'step' cannot be zero");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

struct RangeParams
{
    double start;
    double stop;
    double step;
    PartialShape expected_shape;
};

template <typename T>
void run_range_test(const element::Type& et, const RangeParams& params)
{
    auto start =
        make_shared<op::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.start)});
    auto stop = make_shared<op::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.stop)});
    auto step = make_shared<op::Constant>(et, Shape{}, std::vector<T>{static_cast<T>(params.step)});

    auto range = make_shared<op::Range>(start, stop, step);

    EXPECT_EQ(range->get_element_type(), et);
    EXPECT_TRUE(range->get_output_partial_shape(0).same_scheme(params.expected_shape))
        << "Expected shape " << params.expected_shape << " but got "
        << range->get_output_partial_shape(0);
}

struct RangeTest : ::testing::TestWithParam<RangeParams>
{
};

TEST_P(RangeTest, deduce_shape_i8)
{
    run_range_test<int8_t>(element::i8, GetParam());
}

TEST_P(RangeTest, deduce_shape_i16)
{
    run_range_test<int16_t>(element::i16, GetParam());
}

TEST_P(RangeTest, deduce_shape_i32)
{
    run_range_test<int32_t>(element::i32, GetParam());
}

TEST_P(RangeTest, deduce_shape_i64)
{
    run_range_test<int64_t>(element::i64, GetParam());
}

TEST_P(RangeTest, deduce_shape_u8)
{
    run_range_test<uint8_t>(element::u8, GetParam());
}

TEST_P(RangeTest, deduce_shape_u16)
{
    run_range_test<uint16_t>(element::u16, GetParam());
}

TEST_P(RangeTest, deduce_shape_u32)
{
    run_range_test<uint32_t>(element::u32, GetParam());
}

TEST_P(RangeTest, deduce_shape_u64)
{
    run_range_test<uint64_t>(element::u64, GetParam());
}

TEST_P(RangeTest, deduce_shape_bf16)
{
    run_range_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeTest, deduce_shape_f16)
{
    run_range_test<float16>(element::f16, GetParam());
}

TEST_P(RangeTest, deduce_shape_f32)
{
    run_range_test<float>(element::f32, GetParam());
}

TEST_P(RangeTest, deduce_shape_f64)
{
    run_range_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_CASE_P(type_prop,
                        RangeTest,
                        ::testing::Values(RangeParams{0, 5, 1, PartialShape{5}},
                                          RangeParams{0, 22, 2, PartialShape{11}},
                                          RangeParams{1, 23, 2, PartialShape{11}},
                                          RangeParams{1, 22, 2, PartialShape{11}},
                                          RangeParams{0, 0, 1, PartialShape{0}},
                                          RangeParams{1, 0, 2, PartialShape{0}}));

struct RangeTestWithNegatives : ::testing::TestWithParam<RangeParams>
{
};

TEST_P(RangeTestWithNegatives, deduce_shape_i8)
{
    run_range_test<int8_t>(element::i8, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_i16)
{
    run_range_test<int16_t>(element::i16, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_i32)
{
    run_range_test<int32_t>(element::i32, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_i64)
{
    run_range_test<int64_t>(element::i64, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_bf16)
{
    run_range_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_f16)
{
    run_range_test<float16>(element::f16, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_f32)
{
    run_range_test<float>(element::f32, GetParam());
}

TEST_P(RangeTestWithNegatives, deduce_shape_f64)
{
    run_range_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_CASE_P(type_prop,
                        RangeTestWithNegatives,
                        ::testing::Values(RangeParams{2, 0, -2, PartialShape{1}},
                                          RangeParams{2, 0, -1, PartialShape{2}},
                                          RangeParams{-19, 19, 1, PartialShape{38}},
                                          RangeParams{-19, 19, 3, PartialShape{13}},
                                          RangeParams{20, -19, 1, PartialShape{0}}));

struct RangeTestFloating : ::testing::TestWithParam<RangeParams>
{
};

TEST_P(RangeTestFloating, deduce_shape_bf16)
{
    run_range_test<bfloat16>(element::bf16, GetParam());
}

TEST_P(RangeTestFloating, deduce_shape_f16)
{
    run_range_test<float16>(element::f16, GetParam());
}

TEST_P(RangeTestFloating, deduce_shape_f32)
{
    run_range_test<float>(element::f32, GetParam());
}

TEST_P(RangeTestFloating, deduce_shape_f64)
{
    run_range_test<double>(element::f64, GetParam());
}

INSTANTIATE_TEST_CASE_P(type_prop,
                        RangeTestFloating,
                        ::testing::Values(RangeParams{0, 1, 0.25, PartialShape{4}},
                                          RangeParams{-1, 1, 0.25, PartialShape{8}},
                                          RangeParams{-1, 0.875, 0.25, PartialShape{8}}));

TEST(type_prop, rnn_cell)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto rnn_cell = make_shared<op::RNNCell>(X, W, R, H_t, hidden_size);
    EXPECT_EQ(rnn_cell->output(0).get_element_type(), element::f32);
    EXPECT_EQ(rnn_cell->output(0).get_shape(), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, rnn_cell_invalid_input)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<op::Parameter>(element::f32, Shape{2 * hidden_size, input_size});
    try
    {
        const auto rnn_cell = make_shared<op::RNNCell>(X, W, R, H_t, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor W must have shape"));
    }

    // Invalid R tensor shape.
    W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, 1});
    try
    {
        const auto rnn_cell = make_shared<op::RNNCell>(X, W, R, H_t, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor R must have shape"));
    }

    // Invalid H_t tensor shape.
    R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    H_t = make_shared<op::Parameter>(element::f32, Shape{4, hidden_size});
    try
    {
        const auto rnn_cell = make_shared<op::RNNCell>(X, W, R, H_t, hidden_size);
        FAIL() << "RNNCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor H_t must have shape"));
    }

    // Invalid B tensor shape.
    H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto B = make_shared<op::Parameter>(element::f32, Shape{hidden_size});
    try
    {
        const auto rnn_cell = make_shared<op::RNNCell>(X, W, R, H_t, hidden_size, B);
        FAIL() << "RNNCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor B must have shape"));
    }
}

TEST(type_prop, gru_cell)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto gru_cell = make_shared<op::GRUCell>(X, W, R, H_t, hidden_size);
    EXPECT_EQ(gru_cell->output(0).get_element_type(), element::f32);
    EXPECT_EQ(gru_cell->output(0).get_shape(), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, gru_cell_invalid_input)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    try
    {
        const auto gru_cell = make_shared<op::GRUCell>(X, W, R, H_t, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor W must have shape"));
    }

    // Invalid R tensor shape.
    W = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, 1});
    try
    {
        const auto gru_cell = make_shared<op::GRUCell>(X, W, R, H_t, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor R must have shape"));
    }

    // Invalid H_t tensor shape.
    R = make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    H_t = make_shared<op::Parameter>(element::f32, Shape{4, hidden_size});
    try
    {
        const auto gru_cell = make_shared<op::GRUCell>(X, W, R, H_t, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor H_t must have shape"));
    }

    // Invalid B tensor shape.
    H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    auto B = make_shared<op::Parameter>(element::f32, Shape{hidden_size});
    try
    {
        const auto gru_cell = make_shared<op::GRUCell>(X, W, R, H_t, hidden_size, B);
        FAIL() << "GRUCell node was created with invalid data.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input tensor B must have shape"));
    }
}

TEST(type_prop, quantized_conv_8_bit_output)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                            filter,
                                                            strides,
                                                            dilation,
                                                            padding_below,
                                                            padding_above,
                                                            dilation,
                                                            scale,
                                                            u8_zero_point,
                                                            scale,
                                                            i8_zero_point,
                                                            scale,
                                                            i8_zero_point,
                                                            output_type,
                                                            axes,
                                                            axes,
                                                            axes);

    ASSERT_EQ(quant_conv->get_element_type(), output_type);
    ASSERT_EQ(quant_conv->get_shape(), output_shape);
}

TEST(type_prop, quantized_conv_32_bit_output)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type i32 = element::i32;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i32;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                            filter,
                                                            strides,
                                                            dilation,
                                                            padding_below,
                                                            padding_above,
                                                            dilation,
                                                            scale,
                                                            u8_zero_point,
                                                            scale,
                                                            i8_zero_point,
                                                            scale,
                                                            i8_zero_point,
                                                            output_type,
                                                            axes,
                                                            axes,
                                                            axes);

    ASSERT_EQ(quant_conv->get_element_type(), output_type);
    ASSERT_EQ(quant_conv->get_shape(), output_shape);
}

TEST(type_prop, quantized_conv_non_quantized_input_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = f32;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                u8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non-quantized input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input element type (element::Type{32, 1, 1, 0, \"float\"}) "
                             "must be a quantized type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_quantized_filter_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = f32;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                u8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non-quantized filter not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Filter element type (element::Type{32, 1, 1, 0, \"float\"}) "
                             "must be a quantized type");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_dyn_output_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = f32;
    element::Type output_type = element::dynamic;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                u8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use dynamic output type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Output element type must not be dynamic");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_floating_point_scale_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = i8;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto i8_zero_point = make_shared<op::Parameter>(element::i8, Shape{});
    auto u8_zero_point = make_shared<op::Parameter>(element::u8, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                u8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                scale,
                                                                i8_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non floating point scale not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Scale must be a floating point number");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_input_zero_point_type_mismatch_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = i8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use zero point type different from input type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Input Zero point element type (element::Type{8, 0, 1, 1, \"int8_t\"}) must "
            "match input element type (element::Type{8, 0, 0, 1, \"uint8_t\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_filter_zero_point_type_mismatch_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = u8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use zero point type different from filter type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Filter Zero point element type (element::Type{8, 0, 0, 1, \"uint8_t\"}) must "
            "match filter element type (element::Type{8, 0, 1, 1, \"int8_t\"})");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_scalar_input_zero_point_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{1, 2});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non scalar input zero point not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input scale and input zero point shape must be same and 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_scalar_filter_zero_point_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{1, 2});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non scalar filter zero point not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Filter scale and filter zero point shape must be same and 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_scalar_output_zero_point_fails)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{1, 2});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non scalar output zero point not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Output scale and output zero point shape must be same and 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_empty_input_axes)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                AxisSet{1},
                                                                axes,
                                                                axes);
        FAIL() << "Attempt to use non empty input axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input, filter and output AxisSet should be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_empty_filter_axes)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                AxisSet{1},
                                                                axes);
        FAIL() << "Attempt to use non empty filter axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input, filter and output AxisSet should be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, quantized_conv_non_empty_output_axes)
{
    auto strides = Strides{1, 1};
    auto dilation = Strides{1, 1};
    auto padding_below = CoordinateDiff{1, 1};
    auto padding_above = CoordinateDiff{1, 1};
    element::Type f32 = element::f32;
    element::Type i8 = element::i8;
    element::Type u8 = element::u8;
    element::Type input_type = u8;
    element::Type filter_type = i8;
    element::Type output_type = i8;
    element::Type scale_type = f32;
    element::Type input_zero_point_type = u8;
    element::Type filter_zero_point_type = i8;
    element::Type output_zero_point_type = i8;
    Shape output_shape{64, 64, 220, 220};
    AxisSet axes{};

    auto input = make_shared<op::Parameter>(input_type, Shape{64, 3, 224, 224});
    auto filter = make_shared<op::Parameter>(filter_type, Shape{64, 3, 7, 7});
    auto scale = make_shared<op::Parameter>(scale_type, Shape{});
    auto input_zero_point = make_shared<op::Parameter>(input_zero_point_type, Shape{});
    auto filter_zero_point = make_shared<op::Parameter>(filter_zero_point_type, Shape{});
    auto output_zero_point = make_shared<op::Parameter>(output_zero_point_type, Shape{});
    try
    {
        auto quant_conv = make_shared<op::QuantizedConvolution>(input,
                                                                filter,
                                                                strides,
                                                                dilation,
                                                                padding_below,
                                                                padding_above,
                                                                dilation,
                                                                scale,
                                                                input_zero_point,
                                                                scale,
                                                                filter_zero_point,
                                                                scale,
                                                                output_zero_point,
                                                                output_type,
                                                                axes,
                                                                axes,
                                                                AxisSet{1});
        FAIL() << "Attempt to use non empty output axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Input, filter and output AxisSet should be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
