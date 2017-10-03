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

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/assign_tensors.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/propagate_types.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/log.hpp"

#include "test_tools.hpp"

using namespace std;
using namespace ngraph;
namespace ng = ngraph;

TEST(pass, liveness)
{
    string image = "liveness.png";
    string dump_file = "liveness.txt";
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::VisualizeTree>(image);
    pass_manager.register_pass<pass::TopologicalSort>();
    pass_manager.register_pass<pass::PropagateTypes>();
    pass_manager.register_pass<pass::AssignTensors>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.register_pass<pass::DumpSorted>(dump_file);

    shared_ptr<Function>   func      = make_test_graph();
    pass_manager.run_passes(func.get());
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
