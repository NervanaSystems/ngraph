// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
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

// Standalone Goole Test example for ngraph.
// compile and test as follows.
// g++ -std=c++11 simple_gtest2.cc -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -lngraph -lpthread -lgtest -o /tmp/test
// env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib /tmp/test
#include <gtest/gtest.h>
#include "nutils.h"
using namespace std;
using namespace ngraph;

TEST(test, SequenceMask)
{
    // sequences to replace
    vector<size_t> sr{2, 3};
    // mask value
    float mask_val = 1.0;
    // sequences, batches, features...
    Shape input_shape{3, 2, 3};
    auto A = make_shared<op::Parameter>(element::f32, input_shape);
    
    auto mask_shape = input_shape;
    mask_shape[0] = mask_shape[1] = 1;
    auto C = op::Constant::create(element::f32, Shape{}, {mask_val});
    shared_ptr<Node> mask_seq{C};
    mask_seq = make_shared<op::Broadcast>(mask_seq, mask_shape, AxisSet{0,1,2});
   
    // replace select seq with mask
    shared_ptr<Node> r{A};
    for (size_t j = 0; j < 2; ++j)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            if (i < sr[j])
                continue;
            r = make_shared<op::ReplaceSlice>(
                r, mask_seq, Coordinate{i, j, 0}, Coordinate{i + 1, j + 1, 3});
        }
    }
    auto f = make_shared<Function>(r, op::Parameters{A});

    CFrame cf;
    TViews inp, out;
    tie(cf, ignore, inp, out) = get_cfio("CPU", f);
    
    copy_data(inp[0], vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    cf->call(inp, out);
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 1, 1, 16, 17, 18}),
              read_vector<float>(out[0]));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
