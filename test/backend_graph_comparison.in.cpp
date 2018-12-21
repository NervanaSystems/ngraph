//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include <cstdlib>
#include <string>
#include "gtest/gtest.h"

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// clang-format off
#define TESTING_${BACKEND_NAME}_BACKEND 1
#ifdef TESTING_${BACKEND_NAME}_BACKEND // avoid macro is not used warning
#endif
// clang-format on

// Currently only used to test GPU backend, but is expected to be useful
// testing other backends (except CPU which is used as the reference backend)
#if defined(TESTING_GPU_BACKEND)
class serialized_graph_files : public ::testing::TestWithParam<string>
{
public:
    void compare_results(NodeVector& result_nodes,
                         vector<shared_ptr<runtime::Tensor>> ref_results,
                         vector<shared_ptr<runtime::Tensor>> bk_results,
                         vector<shared_ptr<runtime::Tensor>> bk_isolated_results)
    {
        for (int i = 0; i < ref_results.size(); ++i)
        {
            const shared_ptr<runtime::Tensor>& ref_data = ref_results.at(i);
            const shared_ptr<runtime::Tensor>& bk_data = bk_results.at(i);
            const shared_ptr<runtime::Tensor>& bk_isolated_data = bk_isolated_results.at(i);

            std::shared_ptr<ngraph::Node> result_node = result_nodes.at(i);
            cout << "Comparing results for " << result_node->get_name() << endl;
            if (result_node->get_arguments().size() > 0)
            {
                cout << "  inputs:" << endl;
                for (auto& p : result_node->get_arguments())
                {
                    cout << "    " << p->get_name() << " " << p->get_element_type() << endl;
                }
            }

            // Ensure the element types and shapes match between reference and backend
            ASSERT_EQ(ref_data->get_element_type(), bk_data->get_element_type());
            ASSERT_EQ(ref_data->get_element_type(), bk_isolated_data->get_element_type());
            ASSERT_EQ(ref_data->get_element_count(), bk_data->get_element_count());
            ASSERT_EQ(ref_data->get_element_count(), bk_isolated_data->get_element_count());
            ASSERT_EQ(ref_data->get_shape(), bk_data->get_shape());
            ASSERT_EQ(ref_data->get_shape(), bk_isolated_data->get_shape());

            cout << "  output type:       " << ref_data->get_element_type() << endl;
            cout << "  output shape:      " << ref_data->get_shape() << endl;
            cout << "  output # elements: " << ref_data->get_element_count() << endl;

            element::Type et = ref_data->get_element_type();
            if (et == element::boolean)
            {
                vector<char> ref_data_vector = read_vector<char>(ref_data);
                vector<char> bk_data_vector = read_vector<char>(bk_data);
                vector<char> bk_isolated_data_vector = read_vector<char>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close<char>(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close<char>(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::f32)
            {
                vector<float> ref_data_vector = read_float_vector(ref_data);
                vector<float> bk_data_vector = read_float_vector(bk_data);
                vector<float> bk_isolated_data_vector = read_float_vector(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close_f(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close_f(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::f64)
            {
                vector<double> ref_data_vector = read_vector<double>(ref_data);
                vector<double> bk_data_vector = read_vector<double>(bk_data);
                vector<double> bk_isolated_data_vector = read_vector<double>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);

                // When testing with original graph dependencies test w/ loose f64 tolerance
                constexpr int tolerance_bits = 30;
                bool all_close_graph =
                    test::all_close_f(ref_data_vector, bk_data_vector, tolerance_bits);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);

                // When testing with isolated graph dependencies test w/ default (tight) f64 tolerance
                bool all_close_isolated =
                    test::all_close_f(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::i8)
            {
                vector<int8_t> ref_data_vector = read_vector<int8_t>(ref_data);
                vector<int8_t> bk_data_vector = read_vector<int8_t>(bk_data);
                vector<int8_t> bk_isolated_data_vector = read_vector<int8_t>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close<int8_t>(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close<int8_t>(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::i16)
            {
                vector<int16_t> ref_data_vector = read_vector<int16_t>(ref_data);
                vector<int16_t> bk_data_vector = read_vector<int16_t>(bk_data);
                vector<int16_t> bk_isolated_data_vector = read_vector<int16_t>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close<int16_t>(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close<int16_t>(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::i32)
            {
                vector<int32_t> ref_data_vector = read_vector<int32_t>(ref_data);
                vector<int32_t> bk_data_vector = read_vector<int32_t>(bk_data);
                vector<int32_t> bk_isolated_data_vector = read_vector<int32_t>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close<int32_t>(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close<int32_t>(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::i64)
            {
                vector<int64_t> ref_data_vector = read_vector<int64_t>(ref_data);
                vector<int64_t> bk_data_vector = read_vector<int64_t>(bk_data);
                vector<int64_t> bk_isolated_data_vector = read_vector<int64_t>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close<int64_t>(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close<int64_t>(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::u8)
            {
                vector<uint8_t> ref_data_vector = read_vector<uint8_t>(ref_data);
                vector<uint8_t> bk_data_vector = read_vector<uint8_t>(bk_data);
                vector<uint8_t> bk_isolated_data_vector = read_vector<uint8_t>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close<uint8_t>(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close<uint8_t>(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::u16)
            {
                vector<uint16_t> ref_data_vector = read_vector<uint16_t>(ref_data);
                vector<uint16_t> bk_data_vector = read_vector<uint16_t>(bk_data);
                vector<uint16_t> bk_isolated_data_vector = read_vector<uint16_t>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close<uint16_t>(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close<uint16_t>(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::u32)
            {
                vector<uint32_t> ref_data_vector = read_vector<uint32_t>(ref_data);
                vector<uint32_t> bk_data_vector = read_vector<uint32_t>(bk_data);
                vector<uint32_t> bk_isolated_data_vector = read_vector<uint32_t>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close<uint32_t>(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close<uint32_t>(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else if (et == element::u64)
            {
                vector<uint64_t> ref_data_vector = read_vector<uint64_t>(ref_data);
                vector<uint64_t> bk_data_vector = read_vector<uint64_t>(bk_data);
                vector<uint64_t> bk_isolated_data_vector = read_vector<uint64_t>(bk_isolated_data);
                cout << "Test backed op run w/ original graph dependencies:" << endl;
                print_results(ref_data_vector, bk_data_vector);
                bool all_close_graph = test::all_close<uint64_t>(ref_data_vector, bk_data_vector);
                cout << "Test backed op run isolated w/ inputs from ref graph run:" << endl;
                print_results(ref_data_vector, bk_isolated_data_vector);
                bool all_close_isolated =
                    test::all_close<uint64_t>(ref_data_vector, bk_isolated_data_vector);
                EXPECT_TRUE(all_close_graph && all_close_isolated);
            }
            else
            {
                throw runtime_error("unsupported type");
            }
        }
    }

protected:
    serialized_graph_files() { file_name = GetParam(); }
    string file_name;
};

NGRAPH_TEST_P(${BACKEND_NAME}, serialized_graph_files, compare_backends_with_graphs)
{
    // Compare against CPU for speed. Running large graphs against the
    // interpreter is too slow.
    auto ref = runtime::Backend::create("CPU");
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    string frozen_graph_path;
    if (file_name[0] != '/')
    {
        // Path is relative to serialized zoo path
        frozen_graph_path = file_util::path_join(SERIALIZED_ZOO, file_name);
    }
    else
    {
        // Full path - useful for temporary testing to narrow down problem
        frozen_graph_path = file_name;
    }

    cout << frozen_graph_path << endl;
    stringstream ss(frozen_graph_path);
    shared_ptr<Function> func = ngraph::deserialize(ss);

    // Collect set of results to run two back ends with intermediate results set as outputs
    NodeVector new_results;
    for (auto n : func->get_ordered_ops())
    {
        // Don't include op::Results otherwise Function c-tor will complain
        if (!n->is_output() && !n->is_parameter() && !n->is_constant() &&
            !(n->get_outputs().size() > 1))
        {
            // place conditionals here if you want to only make certain ops an output/result node
            new_results.push_back(n);
        }
    }

    // Collect set of parameters and results to run test backend with isolated
    // inputs which will be copied from the reference backend outputs
    ParameterVector isolated_parameters = func->get_parameters();
    NodeVector isolated_results;
    unordered_map<ngraph::Node*, std::shared_ptr<ngraph::Node>> isolated_node_to_original_node;
    for (auto n : func->get_ordered_ops())
    {
        // Don't include op::Results otherwise Function c-tor will complain
        if (!n->is_output() && !n->is_parameter() && !n->is_constant() &&
            !(n->get_outputs().size() > 1))
        {
            NodeVector isolated_op_args;
            for (auto arg : n->get_arguments())
            {
                if (!arg->is_output() && !arg->is_parameter() && !arg->is_constant() &&
                    !(arg->get_outputs().size() > 1))
                {
                    // Create new isolated arg which we'll fill in with reference results
                    auto isolated_param =
                        make_shared<op::Parameter>(arg->get_element_type(), arg->get_shape());
                    isolated_op_args.push_back(isolated_param);
                    isolated_parameters.push_back(isolated_param);
                    isolated_node_to_original_node[isolated_param.get()] = arg;
                }
                else
                {
                    // It's not an output in the reference results - use the original arg
                    isolated_op_args.push_back(arg);
                }
            }
            auto isolated_op = n->copy_with_new_args(isolated_op_args);
            isolated_results.push_back(isolated_op);
        }
    }

    // No need to include original results they are subsumed by new_results
    auto new_func = make_shared<Function>(new_results, func->get_parameters());

    auto ref_func = clone_function(*new_func);
    auto bk_func = clone_function(*new_func);

    vector<shared_ptr<runtime::Tensor>> ref_args;
    vector<shared_ptr<runtime::Tensor>> bk_args;
    default_random_engine engine(2112);
    for (shared_ptr<op::Parameter> param : new_func->get_parameters())
    {
        auto data =
            make_shared<ngraph::runtime::HostTensor>(param->get_element_type(), param->get_shape());
        random_init(data.get(), engine);
        auto ref_tensor = ref->create_tensor(param->get_element_type(), param->get_shape());
        auto bk_tensor = backend->create_tensor(param->get_element_type(), param->get_shape());
        ref_tensor->write(
            data->get_data_ptr(), 0, data->get_element_count() * data->get_element_type().size());
        bk_tensor->write(
            data->get_data_ptr(), 0, data->get_element_count() * data->get_element_type().size());
        ref_args.push_back(ref_tensor);
        bk_args.push_back(bk_tensor);
    }

    vector<shared_ptr<runtime::Tensor>> ref_results;
    vector<shared_ptr<runtime::Tensor>> bk_results;
    unordered_map<ngraph::Node*, shared_ptr<runtime::Tensor>> arg_to_ref_result;

    ref_results.reserve(new_results.size());
    bk_results.reserve(new_results.size());

    for (shared_ptr<Node>& out : new_results)
    {
        auto ref_result = ref->create_tensor(out->get_element_type(), out->get_shape());
        ref_results.push_back(ref_result);
        arg_to_ref_result[out.get()] = ref_result;
        auto bk_result = backend->create_tensor(out->get_element_type(), out->get_shape());
        bk_results.push_back(bk_result);
    }

    if (ref_func->get_parameters().size() != ref_args.size())
    {
        throw ngraph::ngraph_error(
            "Number of ref runtime parameters and allocated arguments don't match");
    }
    if (bk_func->get_parameters().size() != bk_args.size())
    {
        throw ngraph::ngraph_error(
            "Number of backend runtime parameters and allocated arguments don't match");
    }
    if (ref_func->get_results().size() != ref_results.size())
    {
        throw ngraph::ngraph_error(
            "Number of ref runtime results and allocated results don't match");
    }
    if (bk_func->get_results().size() != bk_results.size())
    {
        throw ngraph::ngraph_error(
            "Number of backend runtime results and allocated results don't match");
    }
    ref->call_with_validate(ref->compile(ref_func), ref_results, ref_args);
    auto handle = backend->compile(bk_func);
    backend->call_with_validate(handle, bk_results, bk_args);

    // Now create isolated function for backend being tested where each node of the
    // original graph is tested with inputs copied from reference backend rather
    // than original dependencies.
    //auto bk_isolated_func = clone_function(Function(isolated_results, isolated_parameters));
    auto bk_isolated_func = make_shared<Function>(isolated_results, isolated_parameters);

    // We'll leverage the bk_args we already used above (for same original parameter data values)
    // and then tack on new data copied (whenever possible) from the reference backend results.
    auto& isolated_params = bk_isolated_func->get_parameters();
    for (auto parameter_it = isolated_params.begin() + new_func->get_parameters().size();
         parameter_it != isolated_params.end();
         ++parameter_it)
    {
        auto param = *parameter_it;
        bool found_reference_data = false;
        auto original_node_it = isolated_node_to_original_node.find(param.get());
        if (original_node_it != isolated_node_to_original_node.end())
        {
            auto ref_tensor_it = arg_to_ref_result.find(original_node_it->second.get());
            if (ref_tensor_it != arg_to_ref_result.end())
            {
                found_reference_data = true;
                auto ref_tensor = ref_tensor_it->second;
                auto data = make_shared<ngraph::runtime::HostTensor>(param->get_element_type(),
                                                                     param->get_shape());
                auto bk_tensor =
                    backend->create_tensor(param->get_element_type(), param->get_shape());
                size_t size_in_bytes = ref_tensor->get_size_in_bytes();
                ref_tensor->read(data->get_data_ptr(), 0, size_in_bytes);
                bk_tensor->write(data->get_data_ptr(), 0, size_in_bytes);
                bk_args.push_back(bk_tensor);
            }
        }
        ASSERT_TRUE(found_reference_data);
    }
    vector<shared_ptr<runtime::Tensor>> bk_isolated_results;
    bk_isolated_results.reserve(isolated_results.size());
    for (shared_ptr<Node>& out : isolated_results)
    {
        auto bk_result = backend->create_tensor(out->get_element_type(), out->get_shape());
        bk_isolated_results.push_back(bk_result);
    }
    handle = backend->compile(bk_isolated_func);
    backend->call_with_validate(handle, bk_isolated_results, bk_args);

    compare_results(new_results, ref_results, bk_results, bk_isolated_results);
}

// The set of graphs tested is not currently significant. These graphs were
// chosen because they're already availabe and demonstrate the technique.
NGRAPH_INSTANTIATE_TEST_CASE_P(
    ${BACKEND_NAME},
    tf_resnet8_files,
    serialized_graph_files,
    testing::Values("tensorflow/resnet8/"
                    "tf_function_cluster_12[_XlaCompiledKernel=true,_XlaNumConstantArgs=3,_"
                    "XlaNumResourceArgs=0].v23.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_20[_XlaCompiledKernel=true,_XlaNumConstantArgs=3,_"
                    "XlaNumResourceArgs=0].v23.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_22[_XlaCompiledKernel=true,_XlaNumConstantArgs=4,_"
                    "XlaNumResourceArgs=0].v24.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_23[_XlaCompiledKernel=true,_XlaNumConstantArgs=1,_"
                    "XlaNumResourceArgs=0].v296.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_28[_XlaCompiledKernel=true,_XlaNumConstantArgs=0,_"
                    "XlaNumResourceArgs=0].v13.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_4[_XlaCompiledKernel=true,_XlaNumConstantArgs=1,_"
                    "XlaNumResourceArgs=0].v14.json",
                    "tensorflow/resnet8/"
                    "tf_function_cluster_8[_XlaCompiledKernel=true,_XlaNumConstantArgs=2,_"
                    "XlaNumResourceArgs=0].v28.json"));

#endif // skipping tests due to backend
