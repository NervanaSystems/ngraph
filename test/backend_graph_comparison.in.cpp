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

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

// Params to drive concat_vector_large testing variations
class frozen_graph_files : public ::testing::TestWithParam<std::string>
{
public:
    void compare_results(NodeVector& result_nodes,
                         std::vector<std::shared_ptr<runtime::Tensor>> ref_results,
                         std::vector<std::shared_ptr<runtime::Tensor>> bk_results)
    {
        for (int i = 0; i < ref_results.size(); ++i)
        {
            const std::shared_ptr<runtime::Tensor>& ref_data = ref_results.at(i);
            const std::shared_ptr<runtime::Tensor>& bk_data = bk_results.at(i);

            std::cout << "Comparing results for " << result_nodes.at(i)->get_name() << std::endl;
            if (auto node = std::dynamic_pointer_cast<op::GetOutputElement>(result_nodes.at(i)))
            {
                std::cout << "  Parent node: ";
                for (auto& p : node->get_arguments())
                {
                    std::cout << " " << p->get_name() << std::endl;
                    std::cout << "   nargs: " << p->get_arguments().size() << std::endl;
                }
            }

            EXPECT_EQ(ref_data->get_element_type(), bk_data->get_element_type());
            EXPECT_EQ(ref_data->get_element_count(), bk_data->get_element_count());
            EXPECT_EQ(ref_data->get_shape(), bk_data->get_shape());

            element::Type et = ref_data->get_element_type();
            if (et == element::boolean)
            {
                std::vector<char> ref_data_vector = read_vector<char>(ref_data);
                std::vector<char> bk_data_vector = read_vector<char>(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close<char>(ref_data_vector, bk_data_vector));
            }
            else if ((et == element::f32) || (et == element::f64))
            {
                std::vector<float> ref_data_vector = read_float_vector(ref_data);
                std::vector<float> bk_data_vector = read_float_vector(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close_f(ref_data_vector, bk_data_vector));
            }
            else if (et == element::i8)
            {
                std::vector<int8_t> ref_data_vector = read_vector<int8_t>(ref_data);
                std::vector<int8_t> bk_data_vector = read_vector<int8_t>(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close<int8_t>(ref_data_vector, bk_data_vector));
            }
            else if (et == element::i16)
            {
                std::vector<int16_t> ref_data_vector = read_vector<int16_t>(ref_data);
                std::vector<int16_t> bk_data_vector = read_vector<int16_t>(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close<int16_t>(ref_data_vector, bk_data_vector));
            }
            else if (et == element::i32)
            {
                std::vector<int32_t> ref_data_vector = read_vector<int32_t>(ref_data);
                std::vector<int32_t> bk_data_vector = read_vector<int32_t>(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close<int32_t>(ref_data_vector, bk_data_vector));
            }
            else if (et == element::i64)
            {
                std::vector<int64_t> ref_data_vector = read_vector<int64_t>(ref_data);
                std::vector<int64_t> bk_data_vector = read_vector<int64_t>(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close<int64_t>(ref_data_vector, bk_data_vector));
            }
            else if (et == element::u8)
            {
                std::vector<uint8_t> ref_data_vector = read_vector<uint8_t>(ref_data);
                std::vector<uint8_t> bk_data_vector = read_vector<uint8_t>(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close<uint8_t>(ref_data_vector, bk_data_vector));
            }
            else if (et == element::u16)
            {
                std::vector<uint16_t> ref_data_vector = read_vector<uint16_t>(ref_data);
                std::vector<uint16_t> bk_data_vector = read_vector<uint16_t>(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close<uint16_t>(ref_data_vector, bk_data_vector));
            }
            else if (et == element::u32)
            {
                std::vector<uint32_t> ref_data_vector = read_vector<uint32_t>(ref_data);
                std::vector<uint32_t> bk_data_vector = read_vector<uint32_t>(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close<uint32_t>(ref_data_vector, bk_data_vector));
            }
            else if (et == element::u64)
            {
                std::vector<uint64_t> ref_data_vector = read_vector<uint64_t>(ref_data);
                std::vector<uint64_t> bk_data_vector = read_vector<uint64_t>(bk_data);
                print_results(ref_data_vector, bk_data_vector);
                EXPECT_TRUE(test::all_close<uint64_t>(ref_data_vector, bk_data_vector));
            }
            else
            {
                throw std::runtime_error("unsupported type");
            }
        }
    }

protected:
    frozen_graph_files() { file_name = GetParam(); }
    std::string file_name;
};

NGRAPH_TEST_P(${BACKEND_NAME}, frozen_graph_files, compare_backends_with_graphs)
{
    // Compare against CPU for speed. Running large graphs against the
    // interpreter is too slow.
    auto ref = runtime::Backend::create("CPU");
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::string frozen_graph_path;
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

    std::cout << frozen_graph_path << std::endl;
    std::stringstream ss(frozen_graph_path);
    std::shared_ptr<Function> func = ngraph::deserialize(ss);

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

    //no need to include original results they are subsumed by new_results
    auto new_func = std::make_shared<Function>(new_results, func->get_parameters());

    // // uncomment these lines to serialize the new_func for later use
    // // I use this for splicing a small graph out of a larger one
    // string js = serialize(new_func, 4);
    // std::ofstream outfile;
    // outfile.open("conv_bprop_filters.json");
    // outfile << js;
    // outfile.close();
    // if (new_func) exit(0);

    auto ref_func = clone_function(*new_func);
    auto bk_func = clone_function(*new_func);

    std::vector<std::shared_ptr<runtime::Tensor>> ref_args;
    std::vector<std::shared_ptr<runtime::Tensor>> bk_args;
    std::default_random_engine engine(2112);
    for (std::shared_ptr<op::Parameter> param : new_func->get_parameters())
    {
        auto data = std::make_shared<ngraph::runtime::HostTensor>(param->get_element_type(),
                                                                  param->get_shape());
        random_init(data, engine);
        auto ref_tensor = ref->create_tensor(param->get_element_type(), param->get_shape());
        auto bk_tensor = backend->create_tensor(param->get_element_type(), param->get_shape());
        ref_tensor->write(
            data->get_data_ptr(), 0, data->get_element_count() * data->get_element_type().size());
        bk_tensor->write(
            data->get_data_ptr(), 0, data->get_element_count() * data->get_element_type().size());
        ref_args.push_back(ref_tensor);
        bk_args.push_back(bk_tensor);
    }

    std::vector<std::shared_ptr<runtime::Tensor>> ref_results;
    std::vector<std::shared_ptr<runtime::Tensor>> bk_results;

    ref_results.reserve(new_results.size());
    bk_results.reserve(new_results.size());

    for (std::shared_ptr<Node>& out : new_results)
    {
        auto ref_result = ref->create_tensor(out->get_element_type(), out->get_shape());
        ref_results.push_back(ref_result);
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
    ref->call_with_validate(ref_func, ref_results, ref_args);
    backend->call_with_validate(bk_func, bk_results, bk_args);

    compare_results(new_results, ref_results, bk_results);
}

// The set of graphs tested is not currently significant. These graphs were
// chosen because they're already availabe and demonstrate the technique.
NGRAPH_INSTANTIATE_TEST_CASE_P(
    ${BACKEND_NAME},
    tf_resnet8_files,
    frozen_graph_files,
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
