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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, replace_slice_deduce_vector)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{3});
    auto rsl = make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{2}, Coordinate{5});
    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_EQ(rsl->get_output_shape(0), (Shape{6}));
}

TEST(type_prop, replace_slice_deduce_matrix)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{3, 6});
    auto rsl =
        make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_EQ(rsl->get_output_shape(0), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_strided)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3});
    auto rsl = make_shared<op::v0::ReplaceSlice>(
        param0, param1, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 2});
    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_EQ(rsl->get_output_shape(0), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_strided_uneven)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2});
    auto rsl = make_shared<op::v0::ReplaceSlice>(
        param0, param1, Coordinate{2, 1}, Coordinate{5, 7}, Strides{3, 4});
    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_EQ(rsl->get_output_shape(0), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_vector_edge)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{6});
    auto rsl = make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{6});
    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_EQ(rsl->get_output_shape(0), (Shape{6}));
}

TEST(type_prop, replace_slice_deduce_matrix_edge)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto rsl =
        make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 8});
    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_EQ(rsl->get_output_shape(0), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_zero_cols)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 0});
    auto rsl =
        make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 0});
    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_EQ(rsl->get_output_shape(0), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_matrix_zero_zero)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{0, 0});
    auto rsl =
        make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{0, 0});
    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_EQ(rsl->get_output_shape(0), (Shape{6, 8}));
}

TEST(type_prop, replace_slice_deduce_vector_invalid_strides)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{4});
    try
    {
        auto sl = make_shared<op::v0::ReplaceSlice>(
            param0, param1, Coordinate{0}, Coordinate{7}, Strides{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid slice strides not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{0}), upper bounds "
                        "(Coordinate{7}) and strides (Strides{1, 2}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_arg_rank_mismatch)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{3, 6, 5});
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Argument rank mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument ranks do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_arg_element_type_mismatch)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::i32, Shape{3, 6});
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{2, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Argument element type mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_slice_shape_mismatch)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{3, 6});
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{1, 1}, Coordinate{5, 7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Slice shape mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Shape of replacement tensor ({3,6}) does not match the slice shape ({4,6})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_slice_shape_mismatch_strided)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{4, 6});
    try
    {
        auto rsl = make_shared<op::v0::ReplaceSlice>(
            param0, param1, Coordinate{1, 1}, Coordinate{5, 7}, Strides{1, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Slice shape mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Shape of replacement tensor ({4,6}) does not match the slice shape ({4,3})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector_edge_upper_oob)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{7});
    try
    {
        auto rsl = make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{7});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 0 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_edge_upper_oob)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 9});
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{6, 9});
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bound out of range not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 1 is out of range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_vector_lower_above_upper)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{0});
    try
    {
        auto rsl = make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{3}, Coordinate{2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_above_upper)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 0});
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{0, 5}, Coordinate{6, 4});
        // Should have thrown, so fail if it didn't
        FAIL() << "Lower bound above upper not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_missing)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing lower bound coordinate not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{0}), upper bounds "
                        "(Coordinate{5, 5}) and strides (Strides{1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_upper_missing)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, Coordinate{0, 0}, Coordinate{5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Missing upper bound coordinate not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{0, 0}), upper bounds "
                        "(Coordinate{5}) and strides (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_lower_extra)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl = make_shared<op::v0::ReplaceSlice>(
            param0, param1, Coordinate{0, 0, 0}, Coordinate{5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra lower bound coordinate not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks of lower bounds (Coordinate{0, 0, "
                                         "0}), upper bounds (Coordinate{5, 5}) and "
                                         "strides (Strides{1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_deduce_matrix_upper_extra)
{
    auto param0 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 8});
    auto param1 = make_shared<op::v0::Parameter>(element::f32, Shape{6, 6});
    try
    {
        auto rsl = make_shared<op::v0::ReplaceSlice>(
            param0, param1, Coordinate{0, 0}, Coordinate{5, 5, 5});
        // Should have thrown, so fail if it didn't
        FAIL() << "Extra upper bound coordinate not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Ranks of lower bounds (Coordinate{0, 0}), "
                                         "upper bounds (Coordinate{5, 5, 5}) and "
                                         "strides (Strides{1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_partial_input_rank_dynamic_replacement_rank_dynamic_attribs_ok)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    auto rsl =
        make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_TRUE(rsl->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop,
     replace_slice_partial_input_rank_dynamic_replacement_rank_dynamic_attribs_rank_mismatch)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of lower-bounds/upper-bounds/strides ranks not detected (argument "
                  "rank-dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks of lower bounds (Coordinate{1, 2, 3, 4}), upper bounds "
                        "(Coordinate{1, 3, 5}) and strides (Strides{1, 1, 1, 2}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     replace_slice_partial_input_rank_dynamic_replacement_rank_dynamic_attribs_bounds_crossing)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 8};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Crossing lower/upper bounds not detected (argument rank-dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Lower bound for slice is greater than upper bound at axis 3 (lower "
                        "bounds: Coordinate{1, 2, 3, 8}, upper bounds: Coordinate{1, 3, 5, 7})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_partial_input_rank_static_dynamic_replacement_rank_dynamic_ok)
{
    PartialShape input_shape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    auto rsl =
        make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_TRUE(rsl->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop,
     replace_slice_partial_input_rank_static_dynamic_some_dims_known_replacement_rank_dynamic_ok)
{
    PartialShape input_shape{2, 4, 10, Dimension::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    auto rsl =
        make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_TRUE(
        rsl->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 10, Dimension::dynamic()}));
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_static_dynamic_replacement_rank_dynamic_attribs_rank_mismatches_input)
{
    PartialShape input_shape{Dimension::dynamic(),
                             Dimension::dynamic(),
                             Dimension::dynamic(),
                             Dimension::dynamic(),
                             Dimension::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of attrib ranks with arg ranks not detected (argument rank-static "
                  "dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument ranks do not match the rank of the lower bounds "
                                         "(Coordinate{1, 2, 3, 4}), upper bounds (Coordinate{1, 3, "
                                         "5, 7}), and strides (Strides{1, 1, 1, 2})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_static_dynamic_some_dims_known_replacement_rank_dynamic_upper_bounds_oob)
{
    PartialShape input_shape{2, 2, 10, Dimension::dynamic()};
    PartialShape replacement_shape{PartialShape::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Upper bounds out of bounds not detected (argument rank-static dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Upper bound for slice at axis 1 is out of "
                                         "range (upper bounds: Coordinate{1, 3, 5, "
                                         "7}, argument shape: {2,2,10,?})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, replace_slice_partial_input_rank_dynamic_replacement_rank_static_dynamic_ok)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    auto rsl =
        make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_TRUE(rsl->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop,
     replace_slice_partial_input_rank_dynamic_replacement_rank_static_dynamic_some_dims_known_ok)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{0, Dimension::dynamic(), Dimension::dynamic(), 2};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    auto rsl =
        make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);

    ASSERT_EQ(rsl->get_output_element_type(0), element::f32);
    ASSERT_TRUE(rsl->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_dynamic_replacement_rank_static_dynamic_some_dims_known_attribs_mismatch_replacement_shape)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{1, Dimension::dynamic(), Dimension::dynamic(), 2};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of shape inferred from attributes with provided replacement shape not "
                  "detected (rank-dynamic/rank-static dynamic inputs)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shape of replacement tensor ({1,?,?,2}) does not match "
                                         "the slice shape ({0,1,2,2})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_dynamic_replacement_rank_static_dynamic_attribs_rank_mismatches_replacement)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape replacement_shape{Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of attrib ranks with arg ranks not detected (arguments "
                  "rank-dynamic/rank-static "
                  "dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument ranks do not match the rank of the lower bounds "
                                         "(Coordinate{1, 2, 3, 4}), upper bounds (Coordinate{1, 3, "
                                         "5, 7}), and strides (Strides{1, 1, 1, 2})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    replace_slice_partial_input_rank_static_dynamic_replacement_rank_static_dynamic_argument_ranks_mismatch)
{
    PartialShape input_shape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape replacement_shape{Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic(),
                                   Dimension::dynamic()};
    Coordinate lower_bounds{1, 2, 3, 4};
    Coordinate upper_bounds{1, 3, 5, 7};
    Strides strides{1, 1, 1, 2};

    auto param0 = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto param1 = make_shared<op::v0::Parameter>(element::f32, replacement_shape);
    try
    {
        auto rsl =
            make_shared<op::v0::ReplaceSlice>(param0, param1, lower_bounds, upper_bounds, strides);
        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatching input/replacement ranks not detected (arguments both rank-static "
                  "dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument ranks do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
