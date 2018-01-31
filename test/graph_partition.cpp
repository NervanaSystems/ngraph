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

shared_ptr<runtime::Manager> get_cached_manager(const string& name)
{
    static unordered_map<string, shared_ptr<runtime::Manager>> cached_managers = {};
    if (cached_managers.find(name) == cached_managers.end())
    {
        cached_managers[name] = runtime::Manager::get(name);
    }
    return cached_managers.at(name);
}

shared_ptr<runtime::Backend> get_cached_backend(const string& name)
{
    static unordered_map<string, shared_ptr<runtime::Backend>> cached_backends = {};
    if (cached_backends.find(name) == cached_backends.end())
    {
        cached_backends[name] = get_cached_manager(name)->allocate_backend();
    }
    return cached_backends.at(name);
}

// HybridCallFrame servers 2 purposes:
// 1. HybridBackend's main use case is to test device placement and graph partition routines.
// 2. It also shows how glued-hybrid runtime can be built by combining different runtimes.
class HybridCallFrame
{
public:
    HybridCallFrame(
        const vector<shared_ptr<Function>>& funcs,
        const vector<shared_ptr<runtime::CallFrame>>& call_frames,
        const unordered_map<shared_ptr<Node>, shared_ptr<Node>>& map_node_to_source_node,
        const unordered_map<shared_ptr<op::Parameter>, size_t>& map_parameter_to_index,
        const unordered_map<shared_ptr<Node>, size_t>& map_result_to_index)
        : m_funcs(funcs)
        , m_call_frames(call_frames)
        , m_map_node_to_source_node(map_node_to_source_node)
        , m_map_parameter_to_index(map_parameter_to_index)
        , m_map_result_to_index(map_result_to_index)
    {
    }

    void call(const vector<shared_ptr<runtime::TensorView>>& inputs,
              const vector<shared_ptr<runtime::TensorView>>& outputs)
    {
        // Each input or output of a function has a TensorView from a corresponding backend
        unordered_map<shared_ptr<Node>, shared_ptr<runtime::TensorView>> map_node_to_tensor_view;

        // Call call_frames
        for (auto func_idx = 0; func_idx < m_call_frames.size(); func_idx++)
        {
            // Get placement
            auto func = m_funcs[func_idx];
            auto call_frame = m_call_frames[func_idx];
            Placement placement = get_colocated_function_placement(func);
            if (placement == Placement::DEFAULT)
            {
                // All outputs are pass-through parameters, place it to CPU
                placement = Placement::CPU;
            }

            // Get backend
            auto manager = get_cached_manager(placement_to_string(placement));
            auto backend = get_cached_backend(placement_to_string(placement));

            // Prepare input TensorViews
            vector<shared_ptr<runtime::TensorView>> parameter_tensor_views;
            for (auto parameter : func->get_parameters())
            {
                // Allocate backend specific TensorView
                auto tv = backend->make_primary_tensor_view(parameter->get_element_type(),
                                                            parameter->get_shape());

                if (m_map_parameter_to_index.find(parameter) != m_map_parameter_to_index.end())
                {
                    // Copy from HybridCallFrame's input TensorView
                    assert(m_map_node_to_source_node.at(parameter) == parameter);
                    auto input_idx = m_map_parameter_to_index.at(parameter);
                    auto input_tv = inputs[input_idx];
                    copy_data(tv, read_vector<float>(input_tv));
                }
                else
                {
                    // Copy from upstream function's TensorView
                    assert(m_map_node_to_source_node.at(parameter) != parameter);
                    auto source_node = m_map_node_to_source_node.at(parameter);
                    auto source_tv = map_node_to_tensor_view.at(source_node);
                    copy_data(tv, read_vector<float>(source_tv));
                }

                // Store
                map_node_to_tensor_view[parameter] = tv;
                parameter_tensor_views.push_back(tv);
            }

            // Prepare output TensorViews
            vector<shared_ptr<runtime::TensorView>> result_tensor_views;
            for (auto result : func->get_results())
            {
                // Allocate backend specific TensorView
                auto tv = backend->make_primary_tensor_view(result->get_element_type(),
                                                            result->get_shape());
                // Store
                map_node_to_tensor_view[result] = tv;
                result_tensor_views.push_back(tv);
            }

            // Call
            call_frame->call(parameter_tensor_views, result_tensor_views);
        }

        // Copy to HybridCallFrame's output TensorView
        for (auto func : m_funcs)
        {
            for (auto result : func->get_results())
            {
                if (m_map_result_to_index.find(result) != m_map_result_to_index.end())
                {
                    auto backend_tv = map_node_to_tensor_view.at(result);
                    auto output_idx = m_map_result_to_index.at(result);
                    auto output_tv = outputs[output_idx];
                    copy_data(output_tv, read_vector<float>(backend_tv));
                }
            }
        }
    }

protected:
    vector<shared_ptr<Function>> m_funcs;
    vector<shared_ptr<runtime::CallFrame>> m_call_frames;
    unordered_map<shared_ptr<Node>, shared_ptr<Node>> m_map_node_to_source_node;
    unordered_map<shared_ptr<op::Parameter>, size_t> m_map_parameter_to_index;
    unordered_map<shared_ptr<Node>, size_t> m_map_result_to_index;
};

class HybridBackend
{
public:
    shared_ptr<runtime::TensorView> make_primary_tensor_view(const element::Type& element_type,
                                                             const Shape& shape)
    {
        auto rc = make_shared<runtime::HostTensorView>(element_type, shape);
        return dynamic_pointer_cast<runtime::TensorView>(rc);
    }

    // Returns CallFrame directly, simplifies calling process
    shared_ptr<HybridCallFrame> compile(const shared_ptr<Function>& f)
    {
        // Store f's parameter and outputs, used in runtime
        unordered_map<shared_ptr<op::Parameter>, size_t> map_parameter_to_index;
        for (size_t i = 0; i < f->get_parameters().size(); ++i)
        {
            map_parameter_to_index[f->get_parameters().at(i)] = i;
        }
        unordered_map<shared_ptr<Node>, size_t> map_result_to_index;
        for (size_t i = 0; i < f->get_results().size(); ++i)
        {
            map_result_to_index[f->get_results().at(i)] = i;
        }

        // Parameter's source is either itself, or the output node of the upstream function
        unordered_map<shared_ptr<Node>, shared_ptr<Node>> map_node_to_source_node;

        // Split to functions
        vector<shared_ptr<Function>> funcs =
            split_function_by_placement(f, map_node_to_source_node);

        // Make call frames
        vector<shared_ptr<runtime::CallFrame>> call_frames;
        for (auto func : funcs)
        {
            Placement placement = get_colocated_function_placement(func);
            if (placement == Placement::DEFAULT)
            {
                // All outputs are pass-through parameters, place it to CPU
                placement = Placement::CPU;
            }
            auto manager = get_cached_manager(placement_to_string(placement));
            auto backend = get_cached_backend(placement_to_string(placement));
            auto external = manager->compile(func);
            auto call_frame = backend->make_call_frame(external);
            call_frames.push_back(call_frame);
        }

        return make_shared<HybridCallFrame>(funcs,
                                            call_frames,
                                            map_node_to_source_node,
                                            map_parameter_to_index,
                                            map_result_to_index);
    }
};

static function<Placement(shared_ptr<Node>)> int_multiply_others_cpu_policy =
    [](shared_ptr<Node> node) {
        Placement placement;
        string node_op = node->description();
        if (node_op == "Parameter")
        {
            placement = Placement::DEFAULT;
        }
        else if (node_op == "Multiply")
        {
            placement = Placement::INTERPRETER;
        }
        else
        {
            placement = Placement::CPU;
        }
        return placement;
    };

TEST(graph_partition, placement_all_cpu_policy)
{
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> AplusB = A + B;
    std::shared_ptr<Node> AplusBtimesC = AplusB * C;
    std::shared_ptr<Function> f = make_shared<Function>(AplusBtimesC, op::Parameters{A, B, C});

    for (auto node : f->get_ordered_ops())
    {
        EXPECT_EQ(node->get_placement(), Placement::DEFAULT);
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>([](shared_ptr<Node> node) {
        return node->description() == "Parameter" ? Placement::DEFAULT : Placement::CPU;
    });
    pass_manager.run_passes(f);

    for (auto node : f->get_ordered_ops())
    {
        if (node->description() == "Parameter")
        {
            EXPECT_EQ(node->get_placement(), Placement::DEFAULT);
        }
        else
        {
            EXPECT_EQ(node->get_placement(), Placement::CPU);
        }
    }
}

TEST(graph_partition, placement_int_multiply_others_cpu_policy)
{
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> AplusB = A + B;
    std::shared_ptr<Node> AplusBtimesC = AplusB * C;
    std::shared_ptr<Function> f = make_shared<Function>(AplusBtimesC, op::Parameters{A, B, C});

    for (auto node : f->get_ordered_ops())
    {
        EXPECT_EQ(node->get_placement(), Placement::DEFAULT);
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
    pass_manager.run_passes(f);

    for (auto node : f->get_ordered_ops())
    {
        string node_op = node->description();
        if (node_op == "Add")
        {
            EXPECT_EQ(node->get_placement(), Placement::CPU);
        }
        else if (node_op == "Multiply")
        {
            EXPECT_EQ(node->get_placement(), Placement::INTERPRETER);
        }
        else
        {
            EXPECT_EQ(node->get_placement(), Placement::DEFAULT);
        }
    }
}

TEST(graph_partition, parameter_insert_and_call)
{
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> AplusB = A + B;
    std::shared_ptr<Node> AplusBtimesC = AplusB * C;
    std::shared_ptr<Function> f = make_shared<Function>(AplusBtimesC, op::Parameters{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
    pass_manager.run_passes(f);

    // Insert parameter node P
    std::shared_ptr<op::Parameter> P =
        make_shared<op::Parameter>(AplusB->get_output_element_type(0), AplusB->get_output_shape(0));
    insert_parameter_split_between(AplusB, AplusBtimesC, P);

    // Check input / ouput ports
    EXPECT_EQ(AplusBtimesC->get_input_ops().at(0), P);
    EXPECT_EQ(AplusBtimesC->get_input_ops().at(1), C);

    // Create f0, f1
    std::shared_ptr<Function> f0 = make_shared<Function>(AplusB, op::Parameters{A, B});
    std::shared_ptr<Function> f1 = make_shared<Function>(AplusBtimesC, op::Parameters{P, C});

    // Check f0 correctness
    std::list<shared_ptr<Node>> f0_node_list = f0->get_ordered_ops();
    std::vector<shared_ptr<Node>> f0_nodes(std::make_move_iterator(std::begin(f0_node_list)),
                                           std::make_move_iterator(std::end(f0_node_list)));
    EXPECT_EQ(f0_nodes[0]->description(), "Parameter");
    EXPECT_EQ(f0_nodes[1]->description(), "Parameter");
    EXPECT_EQ(f0_nodes[2]->description(), "Add");

    // Check f1 correctness
    std::list<shared_ptr<Node>> f1_node_list = f1->get_ordered_ops();
    std::vector<shared_ptr<Node>> f1_nodes(std::make_move_iterator(std::begin(f1_node_list)),
                                           std::make_move_iterator(std::end(f1_node_list)));
    EXPECT_EQ(f1_nodes[0]->description(), "Parameter");
    EXPECT_EQ(f1_nodes[1]->description(), "Parameter");
    EXPECT_EQ(f1_nodes[2]->description(), "Multiply");

    // Run f0 on CPU
    auto cpu_manager = runtime::Manager::get(placement_to_string(Placement::CPU));
    auto cpu_external = cpu_manager->compile(f0);
    auto cpu_backend = cpu_manager->allocate_backend();

    shared_ptr<runtime::TensorView> a = cpu_backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = cpu_backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> a_plus_b =
        cpu_backend->make_primary_tensor_view(element::f32, shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto cpu_call_frame = cpu_backend->make_call_frame(cpu_external);
    cpu_call_frame->call({a, b}, {a_plus_b});
    EXPECT_EQ(read_vector<float>(a_plus_b),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());

    // Run f1 on INTERPRETER
    auto int_manager = runtime::Manager::get(placement_to_string(Placement::INTERPRETER));
    auto int_external = int_manager->compile(f1);
    auto int_backend = int_manager->allocate_backend();

    shared_ptr<runtime::TensorView> c = int_backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> p = int_backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> result =
        int_backend->make_primary_tensor_view(element::f32, shape);

    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());
    copy_data(p, read_vector<float>(a_plus_b));

    auto int_call_frame = int_backend->make_call_frame(int_external);
    int_call_frame->call({p, c}, {result});

    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());
}

TEST(graph_partition, hybrid_backend_abc)
{
    //   A   B     C
    //    \ /     /
    //     +     /
    //    ---   /
    //    [P]  /
    //      \ /
    //       *
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> AplusB = A + B;
    std::shared_ptr<Node> AplusBtimesC = AplusB * C;
    std::shared_ptr<Function> f = make_shared<Function>(AplusBtimesC, op::Parameters{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
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

TEST(graph_partition, hybrid_backend_abcd)
{
    //   A   B
    //    \ /
    //    E*
    //    ---
    // C  [P]  D
    //  \ / \ /
    //  F+  G+
    //    \ /
    //    H+
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> D = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> E = A * B;
    std::shared_ptr<Node> F = C + E;
    std::shared_ptr<Node> G = E + D;
    std::shared_ptr<Node> H = F + G;
    std::shared_ptr<Function> f = make_shared<Function>(H, op::Parameters{A, B, C, D});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
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

TEST(graph_partition, hybrid_backend_back_and_forth)
{
    //   A   B
    //    \ / \
    //    D*   |
    //      \ /
    //      E+   C
    //        \ /
    //        F*
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> D = A * B;
    std::shared_ptr<Node> E = D + B;
    std::shared_ptr<Node> F = E * C;
    std::shared_ptr<Function> f = make_shared<Function>(F, op::Parameters{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
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

TEST(graph_partition, hybrid_backend_multi_middle_nodes)
{
    //   A   B   C        A   B   C           C
    //    \ / \ / \        \ / \ /             \
    //    D+  E+  |  =>    D+  E+     PD  PE   |
    //      \ / \ /                     \ / \ /
    //      F*  G*                       F*  G*
    //        \ /                         \ /
    //        H+                           H+
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> D = A + B;
    std::shared_ptr<Node> E = B + C;
    std::shared_ptr<Node> F = D * E;
    std::shared_ptr<Node> G = E * C;
    std::shared_ptr<Node> H = F + G;
    std::shared_ptr<Function> f = make_shared<Function>(H, op::Parameters{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
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

TEST(graph_partition, hybrid_backend_pass_through_param)
{
    //   A   B     C
    //    \ / \    |
    //    D*   |   |
    //      \ /   /
    //      E+   F
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> C = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> D = A * B;
    std::shared_ptr<Node> E = D + B;
    std::shared_ptr<Node> F = C;
    std::shared_ptr<Function> func = make_shared<Function>(Nodes{E, F}, op::Parameters{A, B, C});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
    pass_manager.run_passes(func);

    auto backend = make_shared<HybridBackend>();
    auto cf = backend->compile(func);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> e = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> f = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    cf->call({a, b, c}, {e, f});
    EXPECT_EQ(read_vector<float>(e), (test::NDArray<float, 2>({{10, 18}, {28, 40}})).get_vector());
    EXPECT_EQ(read_vector<float>(f), (test::NDArray<float, 2>({{9, 10}, {11, 12}})).get_vector());
}

TEST(graph_partition, hybrid_backend_constant)
{
    //     A   B
    //      \ /
    // D(CPU)+   C(CPU)
    //        \ /
    //         E*(INTERPRETER)
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> C = op::Constant::create(element::f32, shape, {9, 10, 11, 12});
    std::shared_ptr<Node> D = A + B;
    std::shared_ptr<Node> E = D * C;
    std::shared_ptr<Function> func = make_shared<Function>(E, op::Parameters{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
    pass_manager.run_passes(func);

    auto backend = make_shared<HybridBackend>();
    auto cf = backend->compile(func);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> e = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    cf->call({a, b}, {e});
    EXPECT_EQ(read_vector<float>(e),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());
}

TEST(graph_partition, hybrid_backend_no_split)
{
    //     A   B
    //      \ /
    //       +
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> C = A + B;
    std::shared_ptr<Function> func = make_shared<Function>(C, op::Parameters{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
    pass_manager.run_passes(func);

    auto backend = make_shared<HybridBackend>();
    auto cf = backend->compile(func);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    cf->call({a, b}, {c});
    EXPECT_EQ(read_vector<float>(c), (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

TEST(graph_partition, hybrid_backend_no_compute)
{
    //     A  B
    //      \/
    //      /\
    //     C  D
    Shape shape = Shape{2, 2};
    std::shared_ptr<op::Parameter> A = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<op::Parameter> B = make_shared<op::Parameter>(element::f32, shape);
    std::shared_ptr<Node> D = A;
    std::shared_ptr<Node> C = B;
    std::shared_ptr<Function> func = make_shared<Function>(Nodes{C, D}, op::Parameters{A, B});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::AssignPlacement>(int_multiply_others_cpu_policy);
    pass_manager.run_passes(func);

    auto backend = make_shared<HybridBackend>();
    auto cf = backend->compile(func);

    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> d = backend->make_primary_tensor_view(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    cf->call({a, b}, {c, d});
    EXPECT_EQ(read_vector<float>(c), (test::NDArray<float, 2>({{5, 6}, {7, 8}})).get_vector());
    EXPECT_EQ(read_vector<float>(d), (test::NDArray<float, 2>({{1, 2}, {3, 4}})).get_vector());
}
