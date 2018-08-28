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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/gpu/op/lstm.hpp"
#include "ngraph/runtime/gpu/op/rnn.hpp"
#include "ngraph/runtime/gpu/pass/gpu_rnn_fusion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(gpu_fusion, rnn_fprop_1_lstm_cell)
{
    auto src_layer = make_shared<op::Parameter>(element::f32, Shape{10, 100});
    auto src_iter = make_shared<op::Parameter>(element::f32, Shape{10, 100});
    auto state_iter = make_shared<op::Parameter>(element::f32, Shape{10, 100});
    auto weights_layer = make_shared<op::Parameter>(element::f32, Shape{400, 100});
    auto weights_iter = make_shared<op::Parameter>(element::f32, Shape{400, 100});
    auto bias_layer = make_shared<op::Parameter>(element::f32, Shape{400});
    auto bias_iter = make_shared<op::Parameter>(element::f32, Shape{400});

    const int number_of_timesteps = 1;
    const int number_of_gates_per_cell = 4;
    const int src_seq_length = 1;
    const int src_layer_feature_size = 100;
    const int feature_size = 100;
    const int rnn_direction = 1;
    const int num_of_rnn_fused_layer = 1;
    auto rnn_node = make_shared<op::gpu::Rnn>(src_layer,
                                              src_iter,
                                              weights_layer,
                                              weights_iter,
                                              bias_layer,
                                              bias_iter,
                                              state_iter,
                                              number_of_timesteps,
                                              number_of_gates_per_cell,
                                              src_seq_length,
                                              src_layer_feature_size,
                                              feature_size,
                                              rnn_direction,
                                              num_of_rnn_fused_layer);
    auto rnn_ht_output = make_shared<op::GetOutputElement>(rnn_node, 0);
    auto rnn_ct_output = make_shared<op::GetOutputElement>(rnn_node, 1);

    auto func = make_shared<Function>(
        NodeVector{rnn_ht_output, rnn_ct_output},
        op::ParameterVector{
            src_layer, src_iter, weights_layer, weights_iter, bias_layer, bias_iter, state_iter});
    auto backend = runtime::Backend::create("GPU");

    shared_ptr<runtime::TensorView> src_layer_t =
        backend->create_tensor(element::f32, src_layer->get_shape());
    shared_ptr<runtime::TensorView> src_iter_t =
        backend->create_tensor(element::f32, src_iter->get_shape());
    shared_ptr<runtime::TensorView> state_iter_t =
        backend->create_tensor(element::f32, state_iter->get_shape());
    shared_ptr<runtime::TensorView> weights_layer_t =
        backend->create_tensor(element::f32, weights_layer->get_shape());
    shared_ptr<runtime::TensorView> weights_iter_t =
        backend->create_tensor(element::f32, weights_iter->get_shape());
    shared_ptr<runtime::TensorView> bias_layer_t =
        backend->create_tensor(element::f32, bias_layer->get_shape());
    shared_ptr<runtime::TensorView> bias_iter_t =
        backend->create_tensor(element::f32, bias_iter->get_shape());
    shared_ptr<runtime::TensorView> result_ht = backend->create_tensor(element::f32, {10, 100});
    shared_ptr<runtime::TensorView> result_ct =
        backend->create_tensor(element::f32, Shape{10, 100});

    copy_data(src_layer_t, vector<float>(1000, 1));
    copy_data(src_iter_t, vector<float>(1000, 1));
    copy_data(state_iter_t, vector<float>(1000, 1));
    copy_data(weights_layer_t, vector<float>(400 * 100, 1));
    copy_data(weights_iter_t, vector<float>(400 * 100, 1));
    copy_data(bias_layer_t, vector<float>(400, 1));
    copy_data(bias_iter_t, vector<float>(400, 1));

    backend->call_with_validate(func,
                                {result_ht, result_ct},
                                {src_layer_t,
                                 src_iter_t,
                                 weights_layer_t,
                                 weights_iter_t,
                                 bias_layer_t,
                                 bias_iter_t,
                                 state_iter_t});
    vector<float> expected_ht(10 * 100, 0.964028f);
    vector<float> expected_ct;
    for (size_t i = 0; i < 10 * 100; i++)
    {
        if (i < 1000)
        {
            expected_ct.push_back(0.964028f);
        }
        else
        {
            expected_ct.push_back(2.0f);
        }
    }

    EXPECT_TRUE(test::all_close(expected_ht, read_vector<float>(result_ht)));
    EXPECT_TRUE(test::all_close(expected_ct, read_vector<float>(result_ct)));
}

TEST(gpu_fusion, fuse_lstm_cells)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::gpu::pass::LSTMFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/2rnn_layer_3lstm_cell.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    auto lstm_ops = get_ops_of_type<op::gpu::Lstm>(func);
    EXPECT_EQ(lstm_ops.size(), 6);
}

TEST(gpu_fusion, fuse_2_layer_rnn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::gpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::gpu::pass::RNNFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/2rnn_layer_3lstm_cell.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t count = count_ops_of_type<op::gpu::Rnn>(func);
    auto rnn_ops = get_ops_of_type<op::gpu::Rnn>(func);
    EXPECT_EQ(rnn_ops.size(), count);
    for (auto& node : rnn_ops)
    {
        EXPECT_EQ(node->get_num_timesteps(), node->get_src_sequence_length());
    }
}

TEST(gpu_fusion, fuse_1_layer_rnn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::gpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::gpu::pass::RNNFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/1rnn_layer_3lstm_cell.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t count = count_ops_of_type<op::gpu::Rnn>(func);
    auto rnn_ops = get_ops_of_type<op::gpu::Rnn>(func);
    EXPECT_EQ(rnn_ops.size(), 1);
    EXPECT_EQ(rnn_ops.size(), count);
    for (auto& node : rnn_ops)
    {
        EXPECT_EQ(node->get_num_timesteps(), node->get_src_sequence_length());
    }
}

static std::shared_ptr<Function> make_function(const std::string& file_name)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, file_name);
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    return func;
}

TEST(gpu_fusion, rnn_fusion_inter_vs_gpu_1lstm_cell)
{
    const std::string file_name("mxnet/1_lstm_cell_forward.json");
    auto gpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto gpu_results = execute(gpu_f, args, "GPU");
    for (size_t i = 0; i < gpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(gpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(gpu_fusion, rnn_fusion_inter_vs_gpu_1rnn_layer_3lstm_cell)
{
    const std::string file_name("mxnet/1rnn_layer_3lstm_cell.json");
    auto gpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto gpu_results = execute(gpu_f, args, "GPU");
    for (size_t i = 0; i < gpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(gpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(gpu_fusion, rnn_fusion_inter_vs_gpu_2rnn_layer_3lstm_cell)
{
    const std::string file_name("mxnet/2rnn_layer_3lstm_cell.json");
    auto gpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto gpu_results = execute(gpu_f, args, "GPU");
    for (size_t i = 0; i < gpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(gpu_results.at(i), int_results.at(i), 1.0e-4f, 1.0e-4f));
    }
}

TEST(gpu_fusion, fuse_rnn_across_layer)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::gpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::gpu::pass::RNNFusion>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    pass_manager.register_pass<runtime::gpu::pass::MultiLayerRNNFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/2rnn_layer_1timestep.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ref_rnn_count = 1;
    auto rnn_count = count_ops_of_type<op::gpu::Rnn>(func);
    EXPECT_EQ(ref_rnn_count, rnn_count);
}

TEST(gpu_fusion, fuse_rnn_across_2layer_1timestep)
{
    const std::string file_name("mxnet/2rnn_layer_1timestep.json");
    auto gpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(0.0f, 1.0f);
    vector<vector<float>> args;

    for (shared_ptr<op::Parameter> param : int_f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }
    auto int_results = execute(int_f, args, "INTERPRETER");
    auto gpu_results = execute(gpu_f, args, "GPU");

    // TODO (pruthvi): Enable this after fixing failing
    // mxnet rnn unit tests
    // EXPECT_EQ(1, count_ops_of_type<op::gpu::Rnn>(gpu_f));
    for (size_t i = 0; i < gpu_results.size(); i++)
    {
        EXPECT_TRUE(test::all_close(gpu_results.at(1), int_results.at(1), 1.0e-4f, 1.0e-4f));
    }
}
