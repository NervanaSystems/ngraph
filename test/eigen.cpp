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

#include <iostream>

#include "Eigen/Dense"
#include "gtest/gtest.h"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

TEST(eigen, simple)
{
    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    EXPECT_FLOAT_EQ(m(1, 1), 1.5);
}

TEST(eigen, test)
{
    float arg0[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float arg1[] = {1, 2, 3, 4, 5, 6, 7, 8};
    vector<float> out(8);

    Eigen::Map<Eigen::Array<float, 8, 1>, Eigen::Unaligned>(out.data()) =
        Eigen::Map<Eigen::Array<float, 8, 1>, Eigen::Unaligned>(arg0) *
        Eigen::Map<Eigen::Array<float, 8, 1>, Eigen::Unaligned>(arg1);

    cout << "result " << join(out) << endl;

    // EigenArray1d<float>(Multiply_163_0, fmt::V{12}) =
    //     EigenArray1d<float>(Maximum_162_0, fmt::V{12}) *
    //     EigenArray1d<float>(Parameter_158_0, fmt::V{12});
}
