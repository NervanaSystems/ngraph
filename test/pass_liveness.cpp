/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
namespace ng = ngraph;

TEST(liveness, constant)
{
    Shape shape{1};
    auto c = op::Constant::create(element::i32, shape, {5});
    auto f = make_shared<Function>(make_shared<op::Negative>(c), op::ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.run_passes(f);

    auto tmp = f->get_ordered_ops();
    vector<shared_ptr<Node>> sorted{tmp.begin(), tmp.end()};
    ASSERT_EQ(3, sorted.size());
    EXPECT_EQ(0, sorted[0]->liveness_live_list.size());
    EXPECT_EQ(0, sorted[0]->liveness_new_list.size());
    EXPECT_EQ(0, sorted[0]->liveness_free_list.size());

    //op::Negative is live on output to op::Result
    EXPECT_EQ(1, sorted[1]->liveness_live_list.size());
    //op::Negative is new
    EXPECT_EQ(1, sorted[1]->liveness_new_list.size());
    EXPECT_EQ(0, sorted[1]->liveness_free_list.size());

    //op::Negative is live on input to op::Result
    EXPECT_EQ(1, sorted[2]->liveness_live_list.size());
    EXPECT_EQ(0, sorted[2]->liveness_new_list.size());
    //op::Negative is freed
    EXPECT_EQ(1, sorted[2]->liveness_free_list.size());
}

TEST(liveness, liveness)
{
    string image = "liveness.png";
    string dump_file = "liveness.txt";
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::VisualizeTree>(image);
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.register_pass<pass::DumpSorted>(dump_file);

    shared_ptr<Function> func = make_test_graph();
    pass_manager.run_passes(func);
    auto sorted = func->get_ordered_ops();

    // for (const Node* node : sorted)
    // {
    //     NGRAPH_INFO << *node;
    //     for (const descriptor::Tensor* tensor : node->liveness_live_list)
    //     {
    //         NGRAPH_INFO << "    " << *tensor;
    //     }
    // }

    // auto x = ng.variable(axes=[]).named('x');
    // auto y = ng.variable(axes=[]).named('y');
    // auto w1 = ng.variable(axes=[]).named('w1');
    // auto w2 = ng.variable(axes=[]).named('w2');

    // auto x2 = x * w1;
    // auto x3 = (x2 * w2).named('result');
    // auto cost = x3 - y;

    // auto dw1 = ng.deriv(cost, w1);
    // auto dw2 = ng.deriv(cost, w2);

    // auto upd1 = ng.assign(w1, w1 + dw1);
    // auto upd2 = ng.assign(w2, w2 + dw2);
    // auto seq_stuff = ng.sequential([upd1, upd2, x3]);

    // auto exc = ex.executor(seq_stuff);
    // return exc;

    // lg = LivenessGraph(exc.exop.ops)
    // lg.layout_memory()

    // for i, node in enumerate(lg.liveness_nodes):
    //     print i, node

    // for node in lg.liveness_nodes:
    //     for var1 in node.live_list:
    //         assert var1.buffer_pool_offset is not None
    //         for var2 in node.live_list:
    //             if var1 != var2:
    //                 if var1.buffer_pool_offset < var2.buffer_pool_offset:
    //                     assert var1.buffer_pool_offset + var1.size <= var2.buffer_pool_offset
    //                 else:
    //                     assert var2.buffer_pool_offset + var2.size <= var1.buffer_pool_offset

    // // for o in egraph.computations:
    // //     print o.values

    // print("max memory {}".format(lg.memory_footprint()))
    // print("worst case memory {}".format(lg.worst_case_memory_usage()))
    // print("memory efficiency {}".format(lg.memory_efficiency()))
    // // // print lg.liveness_json()
}
