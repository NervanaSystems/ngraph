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

// HybridCallFrame servers 2 purposes:
// 1. HybridBackend's main use case is to test device placement and graph partition routines.
// 2. It also shows how glued-hybrid runtime can be built by combining different runtimes.
//
// By default, HybridBackend operates on INTERPRETER (for example, the primary tensor view is
// INTERPRETER tensor view). It falls back to CPU when requested by placement.
class HybridBackend
{
public:
    HybridBackend(const function<Placement(shared_ptr<Node>)>& placement_policy)
        : m_placement_policy(placement_policy)
    {
    }

    ~HybridBackend() {}
    shared_ptr<runtime::TensorView> create_tensor(const element::Type& element_type,
                                                  const Shape& shape)
    {
        return get_cached_backend(Placement::INTERPRETER)->create_tensor(element_type, shape);
    }

    bool compile(const shared_ptr<Function>& func)
    {
        if (!contains_key(m_function_map, func))
        {
            // Clone function
            FunctionInstance instance;
            instance.m_function = clone_function(*func);

            // Run placement pass
            pass::Manager pass_manager;
            pass_manager.register_pass<pass::AssignPlacement>(int_with_cpu_mul_policy);
            pass_manager.run_passes(instance.m_function);

            // Split function to sub_functions
            tie(instance.m_sub_functions, instance.m_map_parameter_to_result) =
                split_function_by_placement(instance.m_function);
            m_function_map.insert({func, instance});

            // Compile subfunctions in corresponding backends
            for (shared_ptr<Function>& sub_function : instance.m_sub_functions)
            {
                Placement placement = get_colocated_function_placement(sub_function);
                auto backend = get_cached_backend(placement);
                backend->compile(sub_function);
            }
        }
        return true;
    }

    bool call_with_validate(const shared_ptr<Function>& func,
                            const vector<shared_ptr<runtime::TensorView>>& outputs,
                            const vector<shared_ptr<runtime::TensorView>>& inputs)
    {
        // Get FunctionInstance
        bool rc = true;
        auto it = m_function_map.find(func);
        if (it == m_function_map.end())
        {
            compile(func);
            it = m_function_map.find(func);
        }
        if (it == m_function_map.end())
        {
            throw runtime_error("Error constructing backend.");
        }
        FunctionInstance& instance = it->second;

        // Parameter and result node in sub_function maps to one TensorView
        unordered_map<shared_ptr<Node>, shared_ptr<runtime::TensorView>> map_node_to_tensor_view;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            map_node_to_tensor_view[instance.m_function->get_parameters()[i]] = inputs[i];
        }
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            map_node_to_tensor_view[instance.m_function->get_results()[i]] = outputs[i];
        }

        // Call subfunctions
        for (shared_ptr<Function>& sub_function : instance.m_sub_functions)
        {
            // Init backend
            Placement placement = get_colocated_function_placement(sub_function);
            auto backend = get_cached_backend(placement);

            // Prepare parameter TensorViews
            vector<shared_ptr<runtime::TensorView>> parameter_tvs;
            for (auto parameter_node : sub_function->get_parameters())
            {
                if (map_node_to_tensor_view.find(parameter_node) != map_node_to_tensor_view.end())
                {
                    parameter_tvs.push_back(map_node_to_tensor_view.at(parameter_node));
                }
                else
                {
                    auto result_node = instance.m_map_parameter_to_result.at(parameter_node);
                    auto result_tv = map_node_to_tensor_view.at(result_node);
                    auto parameter_tv = backend->create_tensor(parameter_node->get_element_type(),
                                                               parameter_node->get_shape());
                    copy_data(parameter_tv, read_vector<float>(result_tv));
                    map_node_to_tensor_view[parameter_node] = parameter_tv;
                    parameter_tvs.push_back(parameter_tv);
                }
            }

            // Prepare result TensorViews
            vector<shared_ptr<runtime::TensorView>> result_tvs;
            for (auto result_node : sub_function->get_results())
            {
                if (map_node_to_tensor_view.find(result_node) != map_node_to_tensor_view.end())
                {
                    result_tvs.push_back(map_node_to_tensor_view.at(result_node));
                }
                else
                {
                    auto result_tv = backend->create_tensor(result_node->get_element_type(),
                                                            result_node->get_shape());
                    map_node_to_tensor_view[result_node] = result_tv;
                    result_tvs.push_back(result_tv);
                }
            }

            // Call
            backend->call_with_validate(sub_function, result_tvs, parameter_tvs);
        }
        return rc;
    }

protected:
    class FunctionInstance
    {
    public:
        shared_ptr<Function> m_function;
        vector<shared_ptr<Function>> m_sub_functions;
        unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>> m_map_parameter_to_result;
    };

    shared_ptr<runtime::Backend> get_cached_backend(Placement placement)
    {
        if (m_cached_backends.find(placement) == m_cached_backends.end())
        {
            m_cached_backends[placement] = runtime::Backend::create(placement_to_string(placement));
        }
        return m_cached_backends.at(placement);
    }

    map<Placement, shared_ptr<runtime::Backend>> m_cached_backends;
    map<shared_ptr<Function>, FunctionInstance> m_function_map;
    function<Placement(shared_ptr<Node>)> m_placement_policy;
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

#ifdef NGRAPH_CPU_ENABLE
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
    auto f = make_shared<Function>(ResultVector{R}, op::ParameterVector{A, B, C});

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
    auto int_backend = runtime::Backend::create(placement_to_string(Placement::INTERPRETER));
    auto cpu_backend = runtime::Backend::create(placement_to_string(Placement::CPU));

    // f0 on INT
    auto a = int_backend->create_tensor(element::f32, shape);
    auto b = int_backend->create_tensor(element::f32, shape);
    auto c = int_backend->create_tensor(element::f32, shape);
    auto r0 = int_backend->create_tensor(element::f32, shape);
    auto r1 = int_backend->create_tensor(element::f32, shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    auto f0 = make_shared<Function>(ResultVector{R0, R1}, op::ParameterVector{A, B, C});
    int_backend->compile(f0);
    int_backend->call_with_validate(f0, {r0, r1}, {a, b, c});

    // f1 on CPU
    auto p0 = cpu_backend->create_tensor(element::f32, shape);
    auto p1 = cpu_backend->create_tensor(element::f32, shape);
    auto r2 = cpu_backend->create_tensor(element::f32, shape);
    copy_data(p0, read_vector<float>(r0));
    copy_data(p1, read_vector<float>(r1));

    auto f1 = make_shared<Function>(ResultVector{R2}, op::ParameterVector{P0, P1});
    cpu_backend->compile(f1);
    cpu_backend->call_with_validate(f1, {r2}, {p0, p1});

    // f2 on INT
    auto p2 = int_backend->create_tensor(element::f32, shape);
    auto r = int_backend->create_tensor(element::f32, shape);
    copy_data(p2, read_vector<float>(r2));

    auto f2 = make_shared<Function>(ResultVector{R}, op::ParameterVector{P2});
    int_backend->compile(f2);
    int_backend->call_with_validate(f2, {r}, {p2});

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
    auto f = make_shared<Function>(ResultVector{R}, op::ParameterVector{A, B, C});

    auto backend = make_shared<HybridBackend>(int_with_cpu_mul_policy);
    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> r = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call_with_validate(f, {r}, {a, b, c});
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

    auto backend = make_shared<HybridBackend>(int_with_cpu_mul_policy);
    backend->compile(f);

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> d = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> r = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());
    copy_data(d, test::NDArray<float, 2>({{13, 14}, {15, 16}}).get_vector());

    backend->call_with_validate(f, {r}, {a, b, c, d});
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

    auto backend = make_shared<HybridBackend>(int_with_cpu_mul_policy);
    backend->compile(f);

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> r = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call_with_validate(f, {r}, {a, b, c});
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

    auto backend = make_shared<HybridBackend>(int_with_cpu_mul_policy);
    backend->compile(f);

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> r = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    backend->call_with_validate(f, {r}, {a, b, c});
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

    auto backend = make_shared<HybridBackend>(int_with_cpu_mul_policy);
    backend->compile(f);

    shared_ptr<runtime::TensorView> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::TensorView> c = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {c}, {a, b});
    EXPECT_EQ(read_vector<float>(c), (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

#endif
