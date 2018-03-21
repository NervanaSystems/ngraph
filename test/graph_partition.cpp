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
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/graph_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/assign_placement.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/util.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static shared_ptr<runtime::Manager> get_cached_manager(const string& name)
{
    static unordered_map<string, shared_ptr<runtime::Manager>> cached_managers = {};
    if (cached_managers.find(name) == cached_managers.end())
    {
        cached_managers[name] = runtime::Manager::get(name);
    }
    return cached_managers.at(name);
}

static shared_ptr<runtime::Backend> get_cached_backend(const string& name)
{
    static unordered_map<string, shared_ptr<runtime::Backend>> cached_backends = {};
    if (cached_backends.find(name) == cached_backends.end())
    {
        cached_backends[name] = get_cached_manager(name)->allocate_backend();
    }
    return cached_backends.at(name);
}

class HybridCallFrame
{
public:
    HybridCallFrame(const vector<shared_ptr<Function>>& sub_functions,
                    const vector<shared_ptr<runtime::CallFrame>>& call_frames,
                    const unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>>&
                        map_parameter_to_result,
                    const vector<shared_ptr<op::Parameter>>& main_function_parameters,
                    const vector<shared_ptr<op::Result>>& main_function_results)
        : m_sub_functions(sub_functions)
        , m_call_frames(call_frames)
        , m_map_parameter_to_result(map_parameter_to_result)
        , m_main_function_parameters(main_function_parameters)
        , m_main_function_results(main_function_results)
    {
    }

    void call(const vector<shared_ptr<runtime::TensorView>>& inputs,
              const vector<shared_ptr<runtime::TensorView>>& outputs)
    {
        // Every parameter and result node in every sub_function maps to one TensorView
        unordered_map<shared_ptr<Node>, shared_ptr<runtime::TensorView>> map_node_to_tensor_view;

        // Main function's parameters and results
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            map_node_to_tensor_view[m_main_function_parameters[i]] = inputs[i];
        }
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            map_node_to_tensor_view[m_main_function_results[i]] = outputs[i];
        }

        // Call call_frames
        for (auto idx = 0; idx < m_call_frames.size(); idx++)
        {
            // Get placement
            auto sub_function = m_sub_functions[idx];
            auto call_frame = m_call_frames[idx];
            Placement placement = get_colocated_function_placement(sub_function);
            if (placement != Placement::CPU && placement != Placement::INTERPRETER)
            {
                throw ngraph_error("Placement must be CPU or INTERPRETER");
            }

            // Get backend
            auto manager = get_cached_manager(placement_to_string(placement));
            auto backend = get_cached_backend(placement_to_string(placement));

            // Prepare input TensorViews
            vector<shared_ptr<runtime::TensorView>> parameter_tvs;
            for (auto parameter_node : sub_function->get_parameters())
            {
                if (map_node_to_tensor_view.find(parameter_node) != map_node_to_tensor_view.end())
                {
                    parameter_tvs.push_back(map_node_to_tensor_view.at(parameter_node));
                }
                else
                {
                    auto result_node = m_map_parameter_to_result.at(parameter_node);
                    auto result_tv = map_node_to_tensor_view.at(result_node);
                    auto parameter_tv = backend->make_primary_tensor_view(
                        parameter_node->get_element_type(), parameter_node->get_shape());
                    copy_data(parameter_tv, read_vector<float>(result_tv));
                    map_node_to_tensor_view[parameter_node] = parameter_tv;
                    parameter_tvs.push_back(parameter_tv);
                }
            }

            // Prepare output TensorViews
            vector<shared_ptr<runtime::TensorView>> result_tvs;
            for (auto result_node : sub_function->get_results())
            {
                if (map_node_to_tensor_view.find(result_node) != map_node_to_tensor_view.end())
                {
                    result_tvs.push_back(map_node_to_tensor_view.at(result_node));
                }
                else
                {
                    auto result_tv = backend->make_primary_tensor_view(
                        result_node->get_element_type(), result_node->get_shape());
                    map_node_to_tensor_view[result_node] = result_tv;
                    result_tvs.push_back(result_tv);
                }
            }

            // Call
            call_frame->call(parameter_tvs, result_tvs);
        }
    }

protected:
    vector<shared_ptr<Function>> m_sub_functions;
    vector<shared_ptr<runtime::CallFrame>> m_call_frames;
    unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>> m_map_parameter_to_result;
    vector<shared_ptr<op::Parameter>> m_main_function_parameters;
    vector<shared_ptr<op::Result>> m_main_function_results;
};

// HybridCallFrame servers 2 purposes:
// 1. HybridBackend's main use case is to test device placement and graph partition routines.
// 2. It also shows how glued-hybrid runtime can be built by combining different runtimes.
//
// By default, HybridBackend operates on INTERPRETER (for example, the primary tensor view is
// INTERPRETER tensor view). It falls back to CPU when requested by placement.
class HybridBackend
{
public:
    shared_ptr<runtime::TensorView> make_primary_tensor_view(const element::Type& element_type,
                                                             const Shape& shape)
    {
        return get_cached_backend("INTERPRETER")->make_primary_tensor_view(element_type, shape);
    }

    // Returns CallFrame directly, simplifies calling process
    shared_ptr<HybridCallFrame> compile(const shared_ptr<Function>& main_function)
    {
        // Store main_functions parameter and results, since main_function will be split
        auto main_function_parameters = main_function->get_parameters();
        auto main_function_results = main_function->get_results();

        // Split to functions
        vector<shared_ptr<Function>> sub_functions;
        unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>> map_parameter_to_result;
        tie(sub_functions, map_parameter_to_result) = split_function_by_placement(main_function);

        // Make call frames
        vector<shared_ptr<runtime::CallFrame>> call_frames;
        for (auto sub_function : sub_functions)
        {
            Placement placement = get_colocated_function_placement(sub_function);
            auto manager = get_cached_manager(placement_to_string(placement));
            auto backend = get_cached_backend(placement_to_string(placement));
            auto external = manager->compile(sub_function);
            auto call_frame = backend->make_call_frame(external);
            call_frames.push_back(call_frame);
        }

        return make_shared<HybridCallFrame>(sub_functions,
                                            call_frames,
                                            map_parameter_to_result,
                                            main_function_parameters,
                                            main_function_results);
    }
};

// Perform all operations on INTERPRETER and fallback Multiply to CPU
static function<Placement(shared_ptr<Node>)> int_with_cpu_mul_policy = [](shared_ptr<Node> node) {
    Placement placement;
    string node_op = node->description();
    if (node_op == "Multiply")
    {
        placement = Placement::CPU;
    }
    else
    {
        placement = Placement::INTERPRETER;
    }
    return placement;
};

TEST(graph_partition, placement_all_cpu_policy)
{
    Shape shape = Shape{2, 2};
    shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<Node> AplusB = A + B;
    shared_ptr<Node> AplusBtimesC = AplusB * C;
    shared_ptr<Function> f = make_shared<Function>(AplusBtimesC, op::ParameterVector{A, B, C});

    for (auto node : f->get_ordered_ops())
    {
        EXPECT_EQ(node->get_placement(), Placement::DEFAULT);
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(
        [](shared_ptr<Node> node) { return Placement::CPU; });
    pass_manager.run_passes(f);

    for (auto node : f->get_ordered_ops())
    {
        EXPECT_EQ(node->get_placement(), Placement::CPU);
    }
}

TEST(graph_partition, placement_int_with_cpu_mul_policy)
{
    Shape shape = Shape{2, 2};
    shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<Node> AplusB = A + B;
    shared_ptr<Node> AplusBtimesC = AplusB * C;
    shared_ptr<Function> f = make_shared<Function>(AplusBtimesC, op::ParameterVector{A, B, C});

    for (auto node : f->get_ordered_ops())
    {
        EXPECT_EQ(node->get_placement(), Placement::DEFAULT);
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_with_cpu_mul_policy);
    pass_manager.run_passes(f);

    for (auto node : f->get_ordered_ops())
    {
        string node_op = node->description();
        if (node_op == "Multiply")
        {
            EXPECT_EQ(node->get_placement(), Placement::CPU);
        }
        else
        {
            EXPECT_EQ(node->get_placement(), Placement::INTERPRETER);
        }
    }
}

TEST(graph_partition, hybrid_abc_manual)
{
    // A   B   C    A   B     C
    //  \ /   /      \ /     /
    //   +D  /        +D    /
    //    \ /         |    /
    //     *E         R0  R1  f0(INT)
    //     |       ------------------
    //     R          P0  P1
    //                 \ /
    //                  *E
    //                  |
    //                  R2    f1(CPU)
    //             ------------------
    //                  P2
    //                  |
    //                  R     f2(INT)
    //             ------------------
    Shape shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = A + B;
    auto E = D * C;
    auto R = make_shared<op::Result>(E);
    auto f = make_shared<Function>(R, op::ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_with_cpu_mul_policy);
    pass_manager.run_passes(f);

    // Insert parameter
    auto RP0 = insert_result_parameter_split(D, E);
    shared_ptr<op::Result> R0 = RP0.first;
    shared_ptr<op::Parameter> P0 = RP0.second;
    auto RP1 = insert_result_parameter_split(C, E);
    shared_ptr<op::Result> R1 = RP1.first;
    shared_ptr<op::Parameter> P1 = RP1.second;
    auto RP2 = insert_result_parameter_split(E, R);
    shared_ptr<op::Result> R2 = RP2.first;
    shared_ptr<op::Parameter> P2 = RP2.second;

    // Backends
    auto int_manager = runtime::Manager::get(placement_to_string(Placement::INTERPRETER));
    auto int_backend = int_manager->allocate_backend();
    auto cpu_manager = runtime::Manager::get(placement_to_string(Placement::CPU));
    auto cpu_backend = cpu_manager->allocate_backend();

    // f0 on INT
    auto a = int_backend->make_primary_tensor_view(element::f32, shape);
    auto b = int_backend->make_primary_tensor_view(element::f32, shape);
    auto c = int_backend->make_primary_tensor_view(element::f32, shape);
    auto r0 = int_backend->make_primary_tensor_view(element::f32, shape);
    auto r1 = int_backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto f0 = make_shared<Function>(ResultVector{R0, R1}, op::ParameterVector{A, B, C});
    auto f0_external = int_manager->compile(f0);
    auto f0_call_frame = int_backend->make_call_frame(f0_external);
    f0_call_frame->call({a, b, c}, {r0, r1});

    // f1 on CPU
    auto p0 = cpu_backend->make_primary_tensor_view(element::f32, shape);
    auto p1 = cpu_backend->make_primary_tensor_view(element::f32, shape);
    auto r2 = cpu_backend->make_primary_tensor_view(element::f32, shape);
    copy_data(p0, read_vector<float>(r0));
    copy_data(p1, read_vector<float>(r1));

    auto f1 = make_shared<Function>(ResultVector{R2}, op::ParameterVector{P0, P1});
    auto f1_external = cpu_manager->compile(f1);
    auto f1_call_frame = cpu_backend->make_call_frame(f1_external);
    f1_call_frame->call({p0, p1}, {r2});

    // f2 on INT
    auto p2 = int_backend->make_primary_tensor_view(element::f32, shape);
    auto r = int_backend->make_primary_tensor_view(element::f32, shape);
    copy_data(p2, read_vector<float>(r2));

    auto f2 = make_shared<Function>(ResultVector{R}, op::ParameterVector{P2});
    auto f2_external = int_manager->compile(f2);
    auto f2_call_frame = int_backend->make_call_frame(f2_external);
    f2_call_frame->call({p2}, {r});

    // Check final result on INT
    EXPECT_EQ(read_vector<float>(r),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());
}

TEST(graph_partition, hybrid_abc)
{
    // Same as hybrid_abc_manual, but using the test hybrid transformer
    //
    // A   B   C    A   B     C
    //  \ /   /      \ /     /
    //   +D  /        +D    /
    //    \ /         |    /
    //     *E         R0  R1  f0(INT)
    //     |       ------------------
    //     R          P0  P1
    //                 \ /
    //                  *E
    //                  |
    //                  R2    f1(CPU)
    //             ------------------
    //                  P2
    //                  |
    //                  R     f2(INT)
    //             ------------------
    Shape shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto D = A + B;
    auto E = D * C;
    auto R = make_shared<op::Result>(E);
    auto f = make_shared<Function>(R, op::ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_with_cpu_mul_policy);
    pass_manager.run_passes(f);

    auto backend = make_shared<HybridBackend>();
    auto cf = backend->compile(f);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> r = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    cf->call({a, b, c}, {r});
    EXPECT_EQ(read_vector<float>(r),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());
}

TEST(graph_partition, hybrid_abcd)
{
    //   A   B
    //    \ /
    // C  E*   D
    //  \ / \ /
    //  F+  G+
    //    \ /
    //    H+
    Shape shape = Shape{2, 2};
    shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> D = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<Node> E = A * B;
    shared_ptr<Node> F = C + E;
    shared_ptr<Node> G = E + D;
    shared_ptr<Node> H = F + G;
    shared_ptr<Function> f = make_shared<Function>(H, op::ParameterVector{A, B, C, D});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_with_cpu_mul_policy);
    pass_manager.run_passes(f);

    auto backend = make_shared<HybridBackend>();
    auto cf = backend->compile(f);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> d = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> r = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());
    copy_data(d, test::NDArray<float, 2>({{13, 14}, {15, 16}}).get_vector());

    cf->call({a, b, c, d}, {r});
    EXPECT_EQ(read_vector<float>(r), (test::NDArray<float, 2>({{32, 48}, {68, 92}})).get_vector());
}

TEST(graph_partition, hybrid_back_and_forth)
{
    // A   B
    //  \ / \
    //  D*   |
    //    \ /
    //    E+   C
    //      \ /
    //      F*
    Shape shape = Shape{2, 2};
    shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<Node> D = A * B;
    shared_ptr<Node> E = D + B;
    shared_ptr<Node> F = E * C;
    shared_ptr<Function> f = make_shared<Function>(F, op::ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_with_cpu_mul_policy);
    pass_manager.run_passes(f);

    auto backend = make_shared<HybridBackend>();
    auto cf = backend->compile(f);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> r = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    cf->call({a, b, c}, {r});
    EXPECT_EQ(read_vector<float>(r),
              (test::NDArray<float, 2>({{90, 180}, {308, 480}})).get_vector());
}

TEST(graph_partition, hybrid_multi_middle_nodes)
{
    // A   B   C
    //  \ / \ / \
    //  D+  E+  |
    //    \ / \ /
    //    F*  G*
    //      \ /
    //      H+
    Shape shape = Shape{2, 2};
    shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<Node> D = A + B;
    shared_ptr<Node> E = B + C;
    shared_ptr<Node> F = D * E;
    shared_ptr<Node> G = E * C;
    shared_ptr<Node> H = F + G;
    shared_ptr<Function> f = make_shared<Function>(H, op::ParameterVector{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_with_cpu_mul_policy);
    pass_manager.run_passes(f);

    auto backend = make_shared<HybridBackend>();
    auto cf = backend->compile(f);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> r = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    cf->call({a, b, c}, {r});
    EXPECT_EQ(read_vector<float>(r),
              (test::NDArray<float, 2>({{210, 288}, {378, 480}})).get_vector());
}

TEST(graph_partition, hybrid_no_split)
{
    // A   B
    //  \ /
    //   +
    Shape shape = Shape{2, 2};
    shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    shared_ptr<Node> C = A + B;
    shared_ptr<Function> f = make_shared<Function>(C, op::ParameterVector{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_with_cpu_mul_policy);
    pass_manager.run_passes(f);

    auto backend = make_shared<HybridBackend>();
    auto cf = backend->compile(f);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    cf->call({a, b}, {c});
    EXPECT_EQ(read_vector<float>(c), (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}
