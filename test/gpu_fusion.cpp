//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <algorithm>
#include <cstdio>
#include <cudnn.h>
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

#if CUDNN_VERSION >= 7200
TEST(gpu_fusion, rnn_fprop_1_lstm_cell)
{
    auto src_layer = make_shared<op::Parameter>(element::f32, Shape{10, 100});
    auto src_iter = make_shared<op::Parameter>(element::f32, Shape{10, 100});
    auto params =
        make_shared<op::Parameter>(element::f32, Shape{400 * 100 + 400 * 100 + 400 + 400});
    auto state_iter = make_shared<op::Parameter>(element::f32, Shape{10, 100});

    const int number_of_timesteps = 1;
    const int number_of_gates_per_cell = 4;
    const int src_seq_length = 1;
    const int src_layer_feature_size = 100;
    const int feature_size = 100;
    const int rnn_direction = 1;
    const int num_of_rnn_fused_layer = 1;
    auto rnn_node = make_shared<op::gpu::Rnn>(src_layer,
                                              src_iter,
                                              params,
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

    auto func = make_shared<Function>(NodeVector{rnn_ht_output, rnn_ct_output},
                                      ParameterVector{src_layer, src_iter, params, state_iter});
    auto backend = runtime::Backend::create("GPU");

    shared_ptr<runtime::Tensor> src_layer_t =
        backend->create_tensor(element::f32, src_layer->get_shape());
    shared_ptr<runtime::Tensor> src_iter_t =
        backend->create_tensor(element::f32, src_iter->get_shape());
    shared_ptr<runtime::Tensor> state_iter_t =
        backend->create_tensor(element::f32, state_iter->get_shape());
    shared_ptr<runtime::Tensor> params_t =
        backend->create_tensor(element::f32, params->get_shape());

    shared_ptr<runtime::Tensor> result_ht = backend->create_tensor(element::f32, {10, 100});
    shared_ptr<runtime::Tensor> result_ct = backend->create_tensor(element::f32, Shape{10, 100});

    copy_data(src_layer_t, vector<float>(1000, 1));
    copy_data(src_iter_t, vector<float>(1000, 1));
    copy_data(state_iter_t, vector<float>(1000, 1));
    copy_data(params_t, vector<float>(shape_size(params->get_shape()), 1));

    auto handle = backend->compile(func);
    backend->call_with_validate(
        handle, {result_ht, result_ct}, {src_layer_t, src_iter_t, params_t, state_iter_t});
    vector<float> expected_ht(10 * 100, 0.964028f);
    vector<float> expected_ct;
    for (size_t i = 0; i < 10 * 100; i++)
    {
        expected_ct.push_back(0.964028f);
    }

    EXPECT_TRUE(test::all_close(expected_ht, read_vector<float>(result_ht)));
    EXPECT_TRUE(test::all_close(expected_ct, read_vector<float>(result_ct)));
}
#endif

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
    auto lstm_ops = get_ops_of_type<op::gpu::Rnn>(func);
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

TEST(DISABLED_gpu_fusion, fuse_1_layer_rnn)
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

TEST(gpu_fusion, lstm_analytic)
{
    auto input_xt = std::make_shared<op::Parameter>(element::f32, Shape{1, 1});
    auto weights_i2h = std::make_shared<op::Parameter>(element::f32, Shape{4, 1});
    auto weights_i2h_reshape =
        std::make_shared<op::Reshape>(weights_i2h, AxisVector{1, 0}, Shape{1, 4});
    auto dot_1 = std::make_shared<op::Dot>(input_xt, weights_i2h_reshape);

    auto bias_i2h = std::make_shared<op::Parameter>(element::f32, Shape{4});
    auto broadcast_bias_i2h = std::make_shared<op::Broadcast>(bias_i2h, Shape{1, 4}, AxisSet{0});
    auto add_1 = std::make_shared<op::Add>(dot_1, broadcast_bias_i2h);

    auto h_const = op::Constant::create(element::f32, Shape{}, {1.0});
    auto hidden_ht = std::make_shared<op::Broadcast>(h_const, Shape{1, 1}, AxisSet{0, 1});
    auto weights_h2h = std::make_shared<op::Parameter>(element::f32, Shape{4, 1});
    auto param2_2_reshape =
        std::make_shared<op::Reshape>(weights_h2h, AxisVector{1, 0}, Shape{1, 4});
    auto dot_2 = std::make_shared<op::Dot>(hidden_ht, param2_2_reshape);

    auto bias_h2h = std::make_shared<op::Parameter>(element::f32, Shape{4});
    auto broadcast_bias_h2h = std::make_shared<op::Broadcast>(bias_h2h, Shape{1, 4}, AxisSet{0});
    auto add_2 = std::make_shared<op::Add>(dot_2, broadcast_bias_h2h);

    auto X = std::make_shared<op::Add>(add_2, add_1);
    // construct forget gate
    auto input_slice_0 = std::make_shared<op::Slice>(X, Coordinate{0, 0}, Coordinate{1, 1});
    auto forget_gate = std::make_shared<op::Sigmoid>(input_slice_0);

    //ct-1 -> cell state
    auto c_const = op::Constant::create(element::f32, Shape{}, {-1.0});
    auto ct_1 = std::make_shared<op::Broadcast>(c_const, Shape{1, 1}, AxisSet{0, 1});
    //auto ct_1 = std::make_shared<op::>(element::f32, Shape{10, 100});
    auto multiply_forget_gate_ct_1 = std::make_shared<op::Multiply>(forget_gate, ct_1);

    // construct input gate
    auto input_slice_1 = std::make_shared<op::Slice>(X, Coordinate{0, 1}, Coordinate{1, 2});
    auto input_gate = std::make_shared<op::Sigmoid>(input_slice_1);
    auto input_slice_2 = std::make_shared<op::Slice>(X, Coordinate{0, 2}, Coordinate{1, 3});
    auto tanh_1 = std::make_shared<op::Tanh>(input_slice_2);
    auto multiply_input_gate_tanh_1 = std::make_shared<op::Multiply>(input_gate, tanh_1);

    auto ct = std::make_shared<op::Add>(multiply_forget_gate_ct_1, multiply_input_gate_tanh_1);

    // construct output gate
    auto input_slice_3 = std::make_shared<op::Slice>(X, Coordinate{0, 3}, Coordinate{1, 4});
    auto output_gate = std::make_shared<op::Sigmoid>(input_slice_3);
    auto tanh_2 = std::make_shared<op::Tanh>(ct);
    auto ht = std::make_shared<op::Multiply>(output_gate, tanh_2);

    auto f = make_shared<Function>(
        NodeVector{ht, ct},
        ParameterVector{input_xt, weights_i2h, weights_h2h, bias_i2h, bias_h2h});

    auto backend = runtime::Backend::create("GPU");

    std::shared_ptr<runtime::Tensor> input_xt_t =
        backend->create_tensor(element::f32, input_xt->get_shape());
    copy_data(input_xt_t, std::vector<float>{1.0});

    std::shared_ptr<runtime::Tensor> weights_i2h_t =
        backend->create_tensor(element::f32, weights_i2h->get_shape());
    copy_data(weights_i2h_t, std::vector<float>{-1.0, -1.0, -1.0, -1.0});

    std::shared_ptr<runtime::Tensor> weights_h2h_t =
        backend->create_tensor(element::f32, weights_h2h->get_shape());
    copy_data(weights_h2h_t, std::vector<float>{-1.0, -1.0, -1.0, -1.0});

    std::shared_ptr<runtime::Tensor> bias_i2h_t =
        backend->create_tensor(element::f32, bias_i2h->get_shape());
    copy_data(bias_i2h_t, std::vector<float>{-1.0, -1.0, -1.0, -1.0});

    std::shared_ptr<runtime::Tensor> bias_h2h_t =
        backend->create_tensor(element::f32, bias_h2h->get_shape());
    copy_data(bias_h2h_t, std::vector<float>{-1.0, -1.0, -1.0, -1.0});

    std::shared_ptr<runtime::Tensor> result_ht =
        backend->create_tensor(element::f32, ht->get_shape());
    std::shared_ptr<runtime::Tensor> result_ct =
        backend->create_tensor(element::f32, ct->get_shape());

    auto handle = backend->compile(f);
    backend->call_with_validate(handle,
                                {result_ht, result_ct},
                                {input_xt_t, weights_i2h_t, weights_h2h_t, bias_i2h_t, bias_h2h_t});

    auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    float ct_val = -sig(-4.0f) + sig(-4.0f) * std::tanh(-4.0f);
    float ht_val = sig(-4.0f) * std::tanh(ct_val);

    EXPECT_TRUE(test::all_close(std::vector<float>{ht_val}, read_vector<float>(result_ht)));
    EXPECT_TRUE(test::all_close(std::vector<float>{ct_val}, read_vector<float>(result_ct)));
}

TEST(gpu_fusion, fuse_2_layer_rnn_1lstm_analytic)
{
    auto input_xt = std::make_shared<op::Parameter>(element::f32, Shape{1, 1});
    auto weights_i2h = std::make_shared<op::Parameter>(element::f32, Shape{4, 1});
    auto weights_i2h_reshape =
        std::make_shared<op::Reshape>(weights_i2h, AxisVector{1, 0}, Shape{1, 4});
    auto dot_1 = std::make_shared<op::Dot>(input_xt, weights_i2h_reshape);

    auto bias_i2h = std::make_shared<op::Parameter>(element::f32, Shape{4});
    auto broadcast_bias_i2h = std::make_shared<op::Broadcast>(bias_i2h, Shape{1, 4}, AxisSet{0});
    auto add_1 = std::make_shared<op::Add>(dot_1, broadcast_bias_i2h);

    auto h_const = op::Constant::create(element::f32, Shape{}, {1.0});
    auto hidden_ht = std::make_shared<op::Broadcast>(h_const, Shape{1, 1}, AxisSet{0, 1});
    auto weights_h2h = std::make_shared<op::Parameter>(element::f32, Shape{4, 1});
    auto param2_2_reshape =
        std::make_shared<op::Reshape>(weights_h2h, AxisVector{1, 0}, Shape{1, 4});
    auto dot_2 = std::make_shared<op::Dot>(hidden_ht, param2_2_reshape);

    auto bias_h2h = std::make_shared<op::Parameter>(element::f32, Shape{4});
    auto broadcast_bias_h2h = std::make_shared<op::Broadcast>(bias_h2h, Shape{1, 4}, AxisSet{0});
    auto add_2 = std::make_shared<op::Add>(dot_2, broadcast_bias_h2h);

    auto X = std::make_shared<op::Add>(add_2, add_1);
    // construct forget gate
    auto input_slice_0 = std::make_shared<op::Slice>(X, Coordinate{0, 0}, Coordinate{1, 1});
    auto forget_gate = std::make_shared<op::Sigmoid>(input_slice_0);

    //ct-1 -> cell state
    auto c_const = op::Constant::create(element::f32, Shape{}, {1.0});
    auto ct_1 = std::make_shared<op::Broadcast>(c_const, Shape{1, 1}, AxisSet{0, 1});
    //auto ct_1 = std::make_shared<op::>(element::f32, Shape{10, 100});
    auto multiply_forget_gate_ct_1 = std::make_shared<op::Multiply>(forget_gate, ct_1);

    // construct input gate
    auto input_slice_1 = std::make_shared<op::Slice>(X, Coordinate{0, 1}, Coordinate{1, 2});
    auto input_gate = std::make_shared<op::Sigmoid>(input_slice_1);
    auto input_slice_2 = std::make_shared<op::Slice>(X, Coordinate{0, 2}, Coordinate{1, 3});
    auto tanh_1 = std::make_shared<op::Tanh>(input_slice_2);
    auto multiply_input_gate_tanh_1 = std::make_shared<op::Multiply>(input_gate, tanh_1);

    auto ct = std::make_shared<op::Add>(multiply_forget_gate_ct_1, multiply_input_gate_tanh_1);

    // construct output gate
    auto input_slice_3 = std::make_shared<op::Slice>(X, Coordinate{0, 3}, Coordinate{1, 4});
    auto output_gate = std::make_shared<op::Sigmoid>(input_slice_3);
    auto tanh_2 = std::make_shared<op::Tanh>(ct);
    auto ht = std::make_shared<op::Multiply>(output_gate, tanh_2);

    // next lstm layer
    auto weights_i2h_0 = std::make_shared<op::Parameter>(element::f32, Shape{4, 1});
    auto weights_i2h_0_reshape_0 =
        std::make_shared<op::Reshape>(weights_i2h_0, AxisVector{1, 0}, Shape{1, 4});
    auto dot_1_0 = std::make_shared<op::Dot>(ht, weights_i2h_0_reshape_0);

    auto bias_i2h_0 = std::make_shared<op::Parameter>(element::f32, Shape{4});
    auto broadcast_bias_i2h_0_0 =
        std::make_shared<op::Broadcast>(bias_i2h_0, Shape{1, 4}, AxisSet{0});
    auto add_1_0 = std::make_shared<op::Add>(dot_1_0, broadcast_bias_i2h_0_0);

    auto h_const_0 = op::Constant::create(element::f32, Shape{}, {1.0});
    auto hidden_ht_0 = std::make_shared<op::Broadcast>(h_const_0, Shape{1, 1}, AxisSet{0, 1});
    auto weights_h2h_0 = std::make_shared<op::Parameter>(element::f32, Shape{4, 1});
    auto param2_2_reshape_0 =
        std::make_shared<op::Reshape>(weights_h2h_0, AxisVector{1, 0}, Shape{1, 4});
    auto dot_2_0 = std::make_shared<op::Dot>(hidden_ht_0, param2_2_reshape_0);

    auto bias_h2h_0 = std::make_shared<op::Parameter>(element::f32, Shape{4});
    auto broadcast_bias_h2h_0_0 =
        std::make_shared<op::Broadcast>(bias_h2h_0, Shape{1, 4}, AxisSet{0});
    auto add_2_0 = std::make_shared<op::Add>(dot_2_0, broadcast_bias_h2h_0_0);

    auto X_0 = std::make_shared<op::Add>(add_2_0, add_1_0);
    // construct forget gate
    auto input_slice_0_0 = std::make_shared<op::Slice>(X_0, Coordinate{0, 0}, Coordinate{1, 1});
    auto forget_gate_0 = std::make_shared<op::Sigmoid>(input_slice_0_0);

    //ct-1 -> cell state
    auto c_const_0 = op::Constant::create(element::f32, Shape{}, {1.0});
    auto ct_1_0 = std::make_shared<op::Broadcast>(c_const_0, Shape{1, 1}, AxisSet{0, 1});
    //auto ct_1 = std::make_shared<op::>(element::f32, Shape{10, 100});
    auto multiply_forget_gate_0_ct_1_0 = std::make_shared<op::Multiply>(forget_gate_0, ct_1_0);

    // construct input gate
    auto input_slice_1_0 = std::make_shared<op::Slice>(X_0, Coordinate{0, 1}, Coordinate{1, 2});
    auto input_gate_0 = std::make_shared<op::Sigmoid>(input_slice_1_0);
    auto input_slice_2_0 = std::make_shared<op::Slice>(X_0, Coordinate{0, 2}, Coordinate{1, 3});
    auto tanh_1_0 = std::make_shared<op::Tanh>(input_slice_2_0);
    auto multiply_input_gate_0_tanh_1_0 = std::make_shared<op::Multiply>(input_gate_0, tanh_1_0);

    auto ct_0 =
        std::make_shared<op::Add>(multiply_forget_gate_0_ct_1_0, multiply_input_gate_0_tanh_1_0);

    // construct output gate
    auto input_slice_3_0 = std::make_shared<op::Slice>(X_0, Coordinate{0, 3}, Coordinate{1, 4});
    auto output_gate_0 = std::make_shared<op::Sigmoid>(input_slice_3_0);
    auto tanh_2_0 = std::make_shared<op::Tanh>(ct_0);
    auto ht_0 = std::make_shared<op::Multiply>(output_gate_0, tanh_2_0);

    auto f = make_shared<Function>(NodeVector{ht_0, ct_0},
                                   ParameterVector{input_xt,
                                                   weights_i2h,
                                                   weights_h2h,
                                                   bias_i2h,
                                                   bias_h2h,
                                                   weights_i2h_0,
                                                   weights_h2h_0,
                                                   bias_i2h_0,
                                                   bias_h2h_0});

    auto backend = runtime::Backend::create("GPU");

    auto params = f->get_parameters();
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> arg_tensors;
    for (shared_ptr<op::Parameter> param : params)
    {
        vector<float> tensor_vals(shape_size(param->get_shape()), 1.0f);
        auto tensor = backend->create_tensor(element::f32, param->get_shape());
        copy_data(tensor, tensor_vals);
        arg_tensors.push_back(tensor);
    }

    std::shared_ptr<runtime::Tensor> result_ht =
        backend->create_tensor(element::f32, ht->get_shape());
    std::shared_ptr<runtime::Tensor> result_ct =
        backend->create_tensor(element::f32, ct->get_shape());

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result_ht, result_ct}, arg_tensors);
    //EXPECT_EQ(1, count_ops_of_type<op::gpu::Rnn>(f));

    auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    float kernel = 4.0f;
    float ct_val_first = sig(kernel) + sig(kernel) * std::tanh(kernel);
    float ht_val_first = sig(kernel) * std::tanh(ct_val_first);

    kernel = 3.0f + ht_val_first;
    float ct_val_second = sig(kernel) + sig(kernel) * std::tanh(kernel);
    float ht_val_second = sig(kernel) * std::tanh(ct_val_second);

    EXPECT_TRUE(test::all_close(std::vector<float>{ht_val_second}, read_vector<float>(result_ht)));
    EXPECT_TRUE(test::all_close(std::vector<float>{ct_val_second}, read_vector<float>(result_ct)));
}

TEST(gpu_fusion, rnn_fusion_inter_vs_gpu_1lstm_cell)
{
    const std::string file_name("mxnet/1_lstm_cell_forward.json");
    auto gpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(-10.0f, 10.0f);
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

TEST(DISABLED_gpu_fusion, rnn_fusion_inter_vs_gpu_1rnn_layer_3lstm_cell)
{
    const std::string file_name("mxnet/1rnn_layer_3lstm_cell.json");
    auto gpu_f = make_function(file_name);
    auto int_f = make_function(file_name);
    test::Uniform<float> rng(-10.0f, 10.0f);
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
    test::Uniform<float> rng(-10.0f, 10.0f);
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
        EXPECT_TRUE(test::all_close(gpu_results.at(i), int_results.at(i), 1.0e-3f, 1.0e-3f));
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
    test::Uniform<float> rng(-10.0f, 10.0f);
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
