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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

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
        DynSliceParams({{4}, {1}, {1}, {1}, {0}}, {{-9000}, {-8000}, {2}}, {{}, {}, {}, {}, {}}),
        DynSliceParams({{5}, {1}, {1}, {1}, {0}}, {{3}, {2}, {1}}, {{}, {}, {}, {}, {}}),
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
