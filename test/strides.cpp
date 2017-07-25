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

#include "strides.hpp"

using namespace std;

TEST(strides, scalar_tree_ctor)
{
    {
        ngraph::scalar_tree tree{2, 3, 4};

        stringstream ss;
        {
            ss << tree;
            EXPECT_STREQ("(2, 3, 4)", ss.str().c_str());
        }
    }
    {
        ngraph::scalar_tree tree{{2, 3}, 4};

        stringstream ss;
        {
            ss << tree;
            EXPECT_STREQ("((2, 3), 4)", ss.str().c_str());
        }
    }
    {
        ngraph::scalar_tree tree{{1, 2}, {3, 4}, 5, {6, 7}};

        stringstream ss;
        {
            ss << tree;
            EXPECT_STREQ("((1, 2), (3, 4), 5, (6, 7))", ss.str().c_str());
        }
    }
    {
        ngraph::scalar_tree tree{1, {2, {3, 4}, {5, 6}}, 7};

        stringstream ss;
        {
            ss << tree;
            EXPECT_STREQ("(1, (2, (3, 4), (5, 6)), 7)", ss.str().c_str());
        }
    }
}

TEST(strides, sizes_ctor)
{
    {
        ngraph::tensor_size size{2, 3, 4};

        stringstream ss;
        {
            ss << size;
            EXPECT_STREQ("(2, 3, 4)", ss.str().c_str());
            EXPECT_EQ(element_type_float, size.get_type());
        }
    }
    {
        ngraph::tensor_size size{{2, 3}, 4};

        stringstream ss;
        {
            ss << size;
            EXPECT_STREQ("((2, 3), 4)", ss.str().c_str());
            EXPECT_EQ(element_type_float, size.get_type());
        }
    }
    {
        ngraph::tensor_size size{{1, 2}, {3, 4}, 5, {6, 7}};

        stringstream ss;
        {
            ss << size;
            EXPECT_STREQ("((1, 2), (3, 4), 5, (6, 7))", ss.str().c_str());
            EXPECT_EQ(element_type_float, size.get_type());
        }
    }
    {
        ngraph::tensor_size size{1, {2, {3, 4}, {5, 6}}, 7};

        stringstream ss;
        {
            ss << size;
            EXPECT_STREQ("(1, (2, (3, 4), (5, 6)), 7)", ss.str().c_str());
            EXPECT_EQ(element_type_float, size.get_type());
        }
    }
    {
        ngraph::tensor_size size{{2, 3, 4}, element_type_int32_t};

        stringstream ss;
        {
            ss << size;
            EXPECT_STREQ("(2, 3, 4)", ss.str().c_str());
            EXPECT_EQ(element_type_int32_t, size.get_type());
        }
    }
}

TEST(strides, sizes_copy)
{
    {
        ngraph::tensor_size size{2, 3, 4};
        auto                copy = size;

        stringstream ss;
        ss << copy;
        EXPECT_STREQ("(2, 3, 4)", ss.str().c_str());
        EXPECT_EQ(element_type_float, copy.get_type());
    }
    {
        ngraph::tensor_size size{{2, 3}, 4};
        auto                copy = size;

        stringstream ss;
        ss << copy;
        EXPECT_STREQ("((2, 3), 4)", ss.str().c_str());
        EXPECT_EQ(element_type_float, copy.get_type());
    }
    {
        ngraph::tensor_size size{{1, 2}, {3, 4}, 5, {6, 7}};
        auto                copy = size;

        stringstream ss;
        ss << copy;
        EXPECT_STREQ("((1, 2), (3, 4), 5, (6, 7))", ss.str().c_str());
        EXPECT_EQ(element_type_float, copy.get_type());
    }
    {
        ngraph::tensor_size size{1, {2, {3, 4}, {5, 6}}, 7};
        auto                copy = size;

        stringstream ss;
        ss << copy;
        EXPECT_STREQ("(1, (2, (3, 4), (5, 6)), 7)", ss.str().c_str());
        EXPECT_EQ(element_type_float, copy.get_type());
    }
}

TEST(strides, strides)
{
    {
        ngraph::tensor_size size{2, 3, 4};
        {
            stringstream ss;
            ss << size.strides();
            EXPECT_STREQ("(48, 16, 4)", ss.str().c_str());
        }
    }
    {
        ngraph::tensor_size size{5, 7, 9, 11};
        {
            stringstream ss;
            ss << size.strides();
            EXPECT_STREQ("(2772, 396, 44, 4)", ss.str().c_str());
        }
    }
    {
        ngraph::tensor_size size{{5, 7, 9}, 11};
        {
            stringstream ss;
            ss << size.strides();
            EXPECT_STREQ("(44, 4)", ss.str().c_str());
        }
    }
    {
        ngraph::tensor_size size{{{5, 7}, 9}, 11};
        {
            stringstream ss;
            ss << size.strides();
            EXPECT_STREQ("(44, 4)", ss.str().c_str());
        }
    }
}

TEST(strides, full_strides)
{
    {
        ngraph::tensor_size size{2, 3, 4};
        {
            stringstream ss;
            ss << size.full_strides();
            EXPECT_STREQ("(48, 16, 4)", ss.str().c_str());
        }
    }
    {
        ngraph::tensor_size size{5, 7, 9, 11};
        {
            stringstream ss;
            ss << size.full_strides();
            EXPECT_STREQ("(2772, 396, 44, 4)", ss.str().c_str());
        }
    }
    {
        ngraph::tensor_size size{{5, 7, 9}, 11};
        {
            stringstream ss;
            ss << size.full_strides();
            EXPECT_STREQ("((2772, 396, 44), 4)", ss.str().c_str());
        }
    }
    {
        ngraph::tensor_size size{{{5, 7}, 9}, 11};
        {
            stringstream ss;
            ss << size.full_strides();
            EXPECT_STREQ("(((2772, 396), 44), 4)", ss.str().c_str());
        }
    }
}
