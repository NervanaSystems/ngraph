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

#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

#include "gtest/gtest.h"

#include "transformers/axes.hpp"
#include "transformers/ndarray.hpp"

using namespace std;

// axes for testing
static auto ax_A = make_axis(2, "A");
static auto ax_B = make_axis(3, "B");
static auto ax_C = make_axis(4, "C");

// axes for testing name matching behavior
static auto ax_A_ = make_axis(5, "A");
static auto ax_B_ = make_axis(6, "B");
static auto ax_C_ = make_axis(7, "C");

//=================================================================================================
// random
//     return a random numpy array with dimension and dtype specified by
//     tensor_description.
//
//     Arguments:
//         tensor_description: location of dimension and dtype specifications for
//             returned array.
//=================================================================================================
ngraph::ndarray random(const TensorDescription& td)
{
    ngraph::ndarray result{td.dtype, td.shape()};
    //     return np.random.random(
    //         tensor_description.shape
    //     ).astype(tensor_description.dtype)
    return result;
}

//=================================================================================================
// tensorview
//     Returns a numpy array which whose buffer is nparr using the
//     tensordescription in td
//
//     Arguments:
//         td TensorDescription: the description of the view of the nparr buffer
//         nparr: the memory the np.array should use
//
//     Returns:
//       np.array view of nparr
//=================================================================================================
ngraph::ndarray tensorview(const TensorDescription& td, ngraph::ndarray& nparr)
{
    ngraph::ndarray result{td.dtype, td.shape(), nparr.buffer};
    //     return np.ndarray(
    //         shape=td.shape,
    //         dtype=td.dtype,
    //         buffer=nparr,
    //         offset=td.offset,
    //         strides=td.strides
    //     )
    return result;
}

void compute_eq(const std::vector<Axis>& _lhs, const std::vector<Axis>& _rhs, bool expected)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    bool actual   = lhs_axes == rhs_axes;
    EXPECT_EQ(expected, actual);
}

void compute_ne(const std::vector<Axis>& _lhs, const std::vector<Axis>& _rhs, bool expected)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    bool actual   = lhs_axes != rhs_axes;
    EXPECT_EQ(expected, actual);
}

void compute_add(const std::vector<Axis>& _lhs,
                 const std::vector<Axis>& _rhs,
                 const std::vector<Axis>& _expected,
                 bool                     expect_failure = false)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    Axes expected = make_axes(_expected);
    if (expect_failure)
    {
        EXPECT_THROW((lhs_axes + rhs_axes), std::invalid_argument);
    }
    else
    {
        Axes actual = lhs_axes + rhs_axes;
        EXPECT_EQ(expected, actual);
    }
}

void compute_subtract(const std::vector<Axis>& _lhs,
                      const std::vector<Axis>& _rhs,
                      const std::vector<Axis>& _expected,
                      bool                     expect_failure = false)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    Axes expected = make_axes(_expected);
    Axes actual   = lhs_axes - rhs_axes;
    EXPECT_EQ(expected, actual);
}

void compute_or(const std::vector<Axis>& _lhs,
                const std::vector<Axis>& _rhs,
                const std::vector<Axis>& _expected,
                bool                     expect_failure = false)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    Axes expected = make_axes(_expected);
    Axes actual   = lhs_axes | rhs_axes;
    EXPECT_EQ(expected, actual);
}

void compute_and(const std::vector<Axis>& _lhs,
                 const std::vector<Axis>& _rhs,
                 const std::vector<Axis>& _expected,
                 bool                     expect_failure = false)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    Axes expected = make_axes(_expected);
    Axes actual   = lhs_axes & rhs_axes;
    EXPECT_EQ(expected, actual);
}

void compute_subset(const std::vector<Axis>& _lhs,
                    const std::vector<Axis>& _rhs,
                    bool                     expected = false)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    bool actual   = rhs_axes.is_sub_set(lhs_axes);
    EXPECT_EQ(expected, actual);
}

void compute_superset(const std::vector<Axis>& _lhs,
                      const std::vector<Axis>& _rhs,
                      bool                     expected = false)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    bool actual   = rhs_axes.is_super_set(lhs_axes);
    EXPECT_EQ(expected, actual);
}

void compute_eq_set(const std::vector<Axis>& _lhs,
                    const std::vector<Axis>& _rhs,
                    bool                     expected = false)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    bool actual   = lhs_axes.is_equal_set(rhs_axes);
    EXPECT_EQ(expected, actual);
}

void compute_ne_set(const std::vector<Axis>& _lhs,
                    const std::vector<Axis>& _rhs,
                    bool                     expected = false)
{
    Axes lhs_axes = make_axes(_lhs);
    Axes rhs_axes = make_axes(_rhs);
    bool actual   = lhs_axes.is_not_equal_set(rhs_axes);
    EXPECT_EQ(expected, actual);
}

TEST(axes, eq)
{
    compute_eq({}, {}, true);
    compute_eq({ax_A}, {}, false);
    compute_eq({ax_A, ax_B}, {ax_B, ax_A}, false);
    compute_eq({ax_A, ax_B}, {ax_B_, ax_A_}, false);
    compute_eq({ax_A, ax_B}, {ax_A_, ax_B}, true);
    compute_eq({ax_A, ax_B}, {ax_A_, ax_B_}, true);
}

TEST(axes, ne)
{
    compute_ne({}, {}, false);
    compute_ne({ax_A}, {}, true);
    compute_ne({ax_A, ax_B}, {ax_B, ax_A}, true);
    compute_ne({ax_A, ax_B}, {ax_B_, ax_A_}, true);
    compute_ne({ax_A, ax_B}, {ax_A_, ax_B}, false);
    compute_ne({ax_A, ax_B}, {ax_A_, ax_B_}, false);
}

TEST(axes, add)
{
    compute_add({}, {}, {});
    compute_add({ax_A}, {}, {ax_A});
    compute_add({ax_A_}, {}, {ax_A});
    compute_add({ax_A}, {ax_B}, {ax_A, ax_B});
    compute_add({ax_A_}, {ax_B_}, {ax_A, ax_B});

    // add (list operation, test exception)
    compute_add({ax_A}, {ax_A}, {}, true);
    compute_add({ax_A}, {ax_A_}, {}, true);
    compute_add({ax_A}, {ax_A_, ax_B}, {}, true);
}

TEST(axes, subtract)
{
    compute_subtract({}, {}, {});
    compute_subtract({}, {ax_A}, {});
    compute_subtract({ax_A}, {}, {ax_A});
    compute_subtract({ax_A, ax_B}, {ax_B}, {ax_A});
    compute_subtract({ax_A, ax_B}, {ax_B_}, {ax_A});
    compute_subtract({ax_A, ax_B}, {ax_A}, {ax_B});
    compute_subtract({ax_A, ax_B}, {ax_A_}, {ax_B});
    compute_subtract({ax_A, ax_B}, {ax_B, ax_A}, {});
    compute_subtract({ax_A, ax_B}, {ax_B_, ax_A_}, {});
}

TEST(axes, or)
{
    compute_or({}, {}, {});
    compute_or({}, {ax_A}, {ax_A});
    compute_or({ax_A}, {}, {ax_A});
    compute_or({ax_A}, {ax_B}, {ax_A, ax_B});
    compute_or({ax_A}, {ax_A_}, {ax_A});
    compute_or({ax_A}, {ax_A_}, {ax_A_});
}

TEST(axes, and)
{
    compute_and({}, {}, {});
    compute_and({}, {ax_A}, {});
    compute_and({ax_A}, {}, {});
    compute_and({ax_A}, {ax_B}, {});
    compute_and({ax_A, ax_B}, {ax_B, ax_C}, {ax_B});
    compute_and({ax_A, ax_B_}, {ax_B, ax_C}, {ax_B});
}

TEST(axes, sub_set)
{
    compute_subset({}, {}, true);
    compute_subset({ax_A}, {}, false);
    compute_subset({}, {ax_A}, true);
    compute_subset({ax_A_}, {ax_A}, true);
    compute_subset({ax_A, ax_B}, {ax_B, ax_A}, true);
    compute_subset({ax_A, ax_B}, {ax_B_, ax_A_}, true);
}

TEST(axes, super_set)
{
    compute_superset({}, {}, true);
    compute_superset({ax_A}, {}, true);
    compute_superset({}, {ax_A}, false);
    compute_superset({ax_A_}, {ax_A}, true);
    compute_superset({ax_A, ax_B}, {ax_B, ax_A}, true);
    compute_superset({ax_A, ax_B}, {ax_B_, ax_A_}, true);
}

TEST(axes, eq_set)
{
    compute_eq_set({}, {}, true);
    compute_eq_set({ax_A}, {}, false);
    compute_eq_set({ax_A}, {ax_A}, true);
    compute_eq_set({ax_A}, {ax_A_}, true);
    compute_eq_set({ax_A, ax_B}, {ax_B_, ax_A_}, true);
}

TEST(axes, ne_set)
{
    compute_ne_set({}, {}, false);
    compute_ne_set({ax_A}, {}, true);
    compute_ne_set({ax_A}, {ax_A}, false);
    compute_ne_set({ax_A}, {ax_A_}, false);
    compute_ne_set({ax_A, ax_B}, {ax_B_, ax_A_}, false);
}

TEST(axes, index)
{
    Axis C = make_axis(5, "C");
    Axis H = make_axis(3, "H");
    Axis N = make_axis(7, "N");

    Axes a{C, H, N};
    EXPECT_EQ(5, a[0].length());
    EXPECT_EQ(3, a[1].length());
    EXPECT_EQ(7, a[2].length());

    Axes b{{C, H}, N};
    EXPECT_EQ(15, b[0].length());
    EXPECT_EQ(7, b[1].length());
}

TEST(axes, as_nested_list)
{
    Axis C = make_axis(5);
    Axis H = make_axis(3);
    Axis N = make_axis(7);

    Axes a{C, H, N};
    cout << "a " << a << endl;

    Axes b{{C, H}, N};
    cout << "b " << b << endl;

    FAIL();
}

TEST(axes, flatten)
{
    Axis C = make_axis(5);
    Axis H = make_axis(3);
    Axis N = make_axis(7);

    Axes b{{C, H}, N};
    auto c = b.flatten();
    EXPECT_TRUE(c.is_flattened());
}

TEST(axes, as_flattened_list)
{
    FAIL();
}

// This test just has to compile
TEST(axes, hash_axis)
{
    std::hash<Axis> h1;
    std::hash<Axes> h2;
    (void)h1;
    (void)h2;
    std::unordered_map<Axis, int> m1; // needs operator==
    std::map<Axis, int>           m2; // needs operator<
    m1[ax_A] = 1;
    m2[ax_A] = 1;
}

TEST(axes, hash_axes)
{
    Axes axes = make_axes({ax_A, ax_B});

    std::unordered_map<Axes, int> m1; // needs operator==
    std::map<Axes, int>           m2; // needs operator<
    m1[axes] = 1;
    m2[axes] = 1;
}

TEST(axes, reaxe_0d_to_1d)
{
    TensorDescription td{};
    ngraph::ndarray   x = random(td);

    // create view of x
    // auto btd = td.broadcast({ax_A});
    // auto x_view = tensorview(btd, x);

    //     # set x
    //     x[()] = 3

    //     # setting e also sets x_view
    //     assert x_view.shape == (ax_A.length,)
    //     assert np.all(x_view == 3)
    FAIL();
}

TEST(axes, reaxe_0d_to_2d)
{
    //     td = TensorDescription(axes=())
    //     x = random(td)

    //     x_view = tensorview(td.broadcast([ax_A, ax_B]), x)

    //     # set x
    //     x[()] = 3

    //     assert x_view.shape == (ax_A.length, ax_B.length)
    //     assert np.all(x_view == 3)
    FAIL();
}

//-----------------------------------------------------------------------------------------------
//     tons of tests relating to reaxeing tensors.
//
//     variables names have a postfix integer which represents the dimensionality
//     of the value.  Views have x_y postfix which means they are y dimensional
//     views of x dimensional buffers.
//
//     I started refactoring into smaller pieces as seen in tests above, but
//     stopped ...
//-----------------------------------------------------------------------------------------------
TEST(axes, simple_tensors)
{
    //     # A simple vector
    //     td1 = TensorDescription(axes=[ax_A])
    //     e1 = random(td1)

    //     td2 = TensorDescription(axes=[ax_A, ax_B])
    //     e2 = random(td2)

    //     # Reaxes
    //     e1_1 = tensorview(td1.broadcast([ax_A, ax_B]), e1)
    //     e1_2 = tensorview(td1.broadcast([ax_B, ax_A]), e1)
    //     e1_3 = tensorview(td1.broadcast([(ax_B, ax_C), ax_A]), e1)

    //     e2_1 = tensorview(td2.broadcast([ax_B, ax_A]), e2)
    //     e2_2 = tensorview(td2.broadcast([ax_A, ax_B]), e2)
    //     e2_3 = tensorview(td2.flatten((
    //         FlattenedAxis((ax_A, ax_B)),
    //     )), e2_2)

    //     assert e1_1.shape == (ax_A.length, ax_B.length)
    //     assert e1_2.shape == (ax_B.length, ax_A.length)

    //     for i in range(ax_A.length):
    //         e1_1[i] = i

    //     for i in range(ax_A.length):
    //         assert e1[i] == i
    //         for j in range(ax_B.length):
    //             assert e1_1[i, j] == i
    //             assert e1_2[j, i] == i
    //         for j in range(ax_B.length * ax_C.length):
    //             assert e1_3[j, i] == i

    //     def val2(i, j):
    //         return (i + 1) * (j + 2)

    //     for i in range(ax_A.length):
    //         for j in range(ax_B.length):
    //             e2[i, j] = val2(i, j)

    //     for i in range(ax_A.length):
    //         for j in range(ax_B.length):
    //             assert e2_1[j, i] == val2(i, j)
    //             assert e2_2[i, j] == val2(i, j)
    //             assert e2_3[i * ax_B.length + j] == val2(i, j)
    FAIL();
}

TEST(axes, sliced_axis)
{
    auto a = make_axis(10);
    auto s = slice_axis(a, slice(0, 5));
    EXPECT_EQ(5, s.length());
}

TEST(axes, sliced_axis_invalid)
{
    auto a = make_axis(10);
    auto s = slice_axis(a, slice(5, 0));
    EXPECT_EQ(0, s.length());
}

TEST(axes, sliced_axis_none_end)
{
    auto a = make_axis(10);
    auto s = slice_axis(a, slice(0));
    EXPECT_EQ(10, s.length());
}

TEST(axes, sliced_axis_negative)
{
    auto a = make_axis(10);
    auto s = slice_axis(a, slice(5, 0, -1));
    EXPECT_EQ(5, s.length());
}

TEST(axes, sliced_axis_negative_invalid)
{
    auto a = make_axis(10);
    auto s = slice_axis(a, slice(0, 5, -1));
    EXPECT_EQ(0, s.length());
}

TEST(axes, sliced_axis_flip)
{
    auto a = make_axis(10);
    auto s = slice_axis(a, slice(-1, -1, -1));
    EXPECT_EQ(0, s.length());
}

TEST(axes, sliced_axis_invalid_step)
{
    EXPECT_THROW(slice(0, 5, 2), std::invalid_argument);
}

TEST(axes, sliced_batch_axis)
{
    // slicing a batch axis should result in a batch axis
    auto a = make_axis(10, "N");
    ASSERT_TRUE(a.is_batch());
    auto s = slice_axis(a, slice(0, 5));
    EXPECT_TRUE(s.is_batch());
}

TEST(axes, sliced_recurrent_axis)
{
    // slicing a recurrent axis should result in a recurrent axis
    auto a = make_axis(10, "REC");
    ASSERT_TRUE(a.is_recurrent());
    auto s = slice_axis(a, slice(0, 5));
    EXPECT_TRUE(s.is_recurrent());
}

TEST(axes, duplicate_axis_names)
{
    try
    {
        AxesMap({{"aaa", "zzz"}, {"bbb", "zzz"}, {"ccc", "yyy"}});
        FAIL();
    }
    catch (std::invalid_argument e)
    {
        EXPECT_TRUE(std::string(e.what()).find("aaa") != std::string::npos);
        EXPECT_TRUE(std::string(e.what()).find("bbb") != std::string::npos);
        EXPECT_TRUE(std::string(e.what()).find("zzz") != std::string::npos);
    }
    catch (...)
    {
        FAIL();
    }
}

TEST(axes, invalid_axes_map_message)
{
    try
    {
        AxesMap({{"aaa", "zzz"}, {"bbb", "zzz"}, {"ccc", "yyy"}});
        FAIL();
    }
    catch (std::invalid_argument e)
    {
        EXPECT_TRUE(std::string(e.what()).find("aaa") != std::string::npos);
        EXPECT_TRUE(std::string(e.what()).find("bbb") != std::string::npos);
        EXPECT_TRUE(std::string(e.what()).find("zzz") != std::string::npos);

        EXPECT_FALSE(std::string(e.what()).find("ccc") != std::string::npos);
        EXPECT_FALSE(std::string(e.what()).find("yyy") != std::string::npos);
    }
    catch (...)
    {
        FAIL();
    }
}

TEST(axes, axes_map)
{
    // map from Axes([aaa, bbb]) to Axes([zzz, bbb]) via AxesMap {aaa: zzz}

    auto a = make_axis(10, "aaa");
    auto b = make_axis(10, "bbb");
    auto z = make_axis(10, "zzz");

    // axes_before = ng.make_axes([a, b])
    auto axes_before = make_axes({a, b});
    // axes_after = ng.make_axes([z, b])
    auto axes_after = make_axes({z, b});

    // axes_map = AxesMap({a.name: z.name})
    AxesMap axes_map({a.name, z.name});

    EXPECT_EQ(axes_after, axes_map.map_axes(axes_before));
    // assert axes_after == axes_map.map_axes(axes_before)
}

TEST(axes, axes_map_immutable)
{
    FAIL();
    // axes_map = AxesMap({})

    // with pytest.raises(TypeError):
    //     axes_map["x"] = "y"
}

TEST(axes, axes_map_init_from_axes)
{
    FAIL();
    // axes_map = AxesMap({ng.make_axis(1, name="aaa"): ng.make_axis(1, name="zzz")})

    // assert axes_map["aaa"] == "zzz"
}

TEST(axes, duplicates)
{
    auto a = make_axis(10, "aaa");
    auto b = make_axis(10, "bbb");
    auto z = make_axis(10, "zzz");

    vector<Axis> a1{a, b, z};
    vector<Axis> a2{a, b, b, z};

    auto l1 = duplicates(a1);
    auto l2 = duplicates(a2);

    EXPECT_EQ(0, l1.size());
    ASSERT_EQ(1, l2.size());
    EXPECT_STREQ("bbb", l2[0].c_str());
}

TEST(tensor_description, broadcast)
{
    // TensorDescription td1{};
    // TensorDescription td2 = td1.broadcast({ax_A});
}
