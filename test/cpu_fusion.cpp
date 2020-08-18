//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "misc.hpp"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_mat_mul_transpose.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/conv_fused.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/gelu.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/batch_fusion.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/dropout.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_mat_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_post_layout_optimizations.hpp"
#include "ngraph/runtime/cpu/pass/cpu_rnn_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_workspace_insertion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(cpu_fusion, gemm_pattern)
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_w);
    auto B = make_shared<op::v0::Parameter>(element::f32, shape_x);
    auto C = make_shared<op::v0::Parameter>(element::f32, shape_b);

    auto dot = make_shared<op::v0::Dot>(A, B);
    auto broadcast = make_shared<op::v0::Broadcast>(C, dot->get_output_shape(0), AxisSet{0});
    auto add = dot + broadcast;

    auto W = std::make_shared<pattern::op::Label>(A);
    auto x = std::make_shared<pattern::op::Label>(B);

    auto reshape_pred = [](Output<Node> n) {
        return static_cast<bool>(as_type<op::v0::Reshape>(n.get_node()));
    };

    auto skip_w = std::make_shared<pattern::op::Skip>(W, reshape_pred);
    auto skip_x = std::make_shared<pattern::op::Skip>(x, reshape_pred);

    auto pdot = make_shared<op::v0::Dot>(skip_w, skip_x);
    auto b = std::make_shared<pattern::op::Label>(C);
    auto pbroadcast = make_shared<op::v0::Broadcast>(b, dot->get_output_shape(0), AxisSet{0});
    auto padd = pdot + pbroadcast;

    TestMatcher n;
    ASSERT_TRUE(n.match(padd, add));
    ASSERT_EQ(n.get_pattern_map()[W], A);
    ASSERT_EQ(n.get_pattern_map()[x], B);
    ASSERT_EQ(n.get_pattern_map()[b], C);

    auto reshape_w = make_shared<op::v0::Reshape>(A, AxisVector{1, 0}, W->get_output_shape(0));
    auto reshape_x = make_shared<op::v0::Reshape>(B, AxisVector{1, 0}, x->get_output_shape(0));
    auto re_dot = make_shared<op::v0::Dot>(reshape_w, reshape_x);
    auto re_add = re_dot + broadcast;
    ASSERT_TRUE(n.match(padd, re_add));
    ASSERT_EQ(n.get_pattern_map()[W], A);
    ASSERT_EQ(n.get_pattern_map()[x], B);
    ASSERT_EQ(n.get_pattern_map()[b], C);

    auto cg = make_shared<op::MatmulBias>(
        W, x, C, W->get_output_shape(0), x->get_output_shape(0), false, false, AxisSet{0});
}

TEST(cpu_fusion, cpu_fusion_pass_basic)
{
    Shape shape{};
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_w);
    auto B = make_shared<op::v0::Parameter>(element::f32, shape_x);
    auto C = make_shared<op::v0::Parameter>(element::f32, shape_b);

    auto dot = make_shared<op::v0::Dot>(A, B);
    auto broadcast = make_shared<op::v0::Broadcast>(C, dot->get_output_shape(0), AxisSet{0});
    auto add = dot + broadcast;
    auto graph = make_shared<op::v0::Abs>(add);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(pass::FusionType::REGULAR_FUSIONS);
    auto func = make_shared<Function>(graph, ParameterVector{A, B, C});
    pass_manager.run_passes(func);
    ASSERT_NE(as_type_ptr<op::MatmulBias>(graph->get_argument(0)), nullptr);
}

TEST(cpu_fusion, matmul_f64)
{
    Shape shape{};
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto A = make_shared<op::v0::Parameter>(element::f64, shape_w);
    auto B = make_shared<op::v0::Parameter>(element::f64, shape_x);
    auto C = make_shared<op::v0::Parameter>(element::f64, shape_b);

    auto dot = make_shared<op::v0::Dot>(A, B);
    auto graph = make_shared<op::v0::Abs>(dot);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(pass::FusionType::REGULAR_FUSIONS);
    auto func = make_shared<Function>(graph, ParameterVector{A, B, C});
    pass_manager.run_passes(func);
    ASSERT_NE(as_type_ptr<op::MatmulBias>(graph->get_argument(0)), nullptr);
}

TEST(cpu_fusion, commutative_matmul_bias)
{
    Shape shape{};
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_w);
    auto B = make_shared<op::v0::Parameter>(element::f32, shape_x);
    auto C = make_shared<op::v0::Parameter>(element::f32, shape_b);

    auto dot = make_shared<op::v0::Dot>(A, B);
    auto broadcast = make_shared<op::v0::Broadcast>(C, dot->get_output_shape(0), AxisSet{0});
    auto add = broadcast + dot;
    auto graph = make_shared<op::v0::Abs>(add);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(pass::FusionType::REGULAR_FUSIONS);
    auto func = make_shared<Function>(graph, ParameterVector{A, B, C});
    pass_manager.run_passes(func);
    ASSERT_NE(as_type_ptr<op::MatmulBias>(graph->get_argument(0)), nullptr);
}

TEST(cpu_fusion, cpu_fusion_pass_matmul_bias)
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    Shape shape_b{1};
    auto W = make_shared<op::v0::Parameter>(element::f32, shape_w);
    auto x = make_shared<op::v0::Parameter>(element::f32, shape_x);
    auto b = make_shared<op::v0::Parameter>(element::f32, shape_b);

    auto mmb = std::make_shared<op::MatmulBias>(
        W, x, Output<Node>(), W->get_output_shape(0), x->get_output_shape(0), false, false);
    auto broadcast = std::make_shared<op::v0::Broadcast>(b, mmb->get_output_shape(0), AxisSet{0});
    auto add = mmb + broadcast;

    auto graph = make_shared<op::v0::Abs>(add);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(pass::FusionType::REGULAR_FUSIONS);
    auto func = make_shared<Function>(graph, ParameterVector{W, x, b});
    pass_manager.run_passes(func);
    auto gmm = graph->get_argument(0);
    ASSERT_TRUE(as_type_ptr<op::MatmulBias>(gmm));
    ASSERT_EQ(gmm->get_argument(2), b);
}

TEST(cpu_fusion, cpu_fusion_pass_matmul_no_bias)
{
    Shape shape_w{4, 2};
    Shape shape_x{1, 4};
    auto W = make_shared<op::v0::Parameter>(element::f32, shape_w);
    auto x = make_shared<op::v0::Parameter>(element::f32, shape_x);

    auto reshape_w = std::make_shared<op::v0::Reshape>(W, AxisVector{1, 0}, Shape{2, 4});
    auto reshape_x = std::make_shared<op::v0::Reshape>(x, AxisVector{1, 0}, Shape{4, 1});
    auto re_dot = make_shared<op::v0::Dot>(reshape_w, reshape_x);
    auto graph = make_shared<op::v0::Abs>(re_dot);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(pass::FusionType::REGULAR_FUSIONS);
    auto func = make_shared<Function>(graph, ParameterVector{W, x});
    pass_manager.run_passes(func);
    size_t mmb = count_ops_of_type<op::MatmulBias>(func);
    ASSERT_EQ(mmb, 1);
}

TEST(cpu_fusion, conv_bias_bprop)
{
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<op::v0::Parameter>(element::f32, shape);
    auto filters = std::make_shared<op::v0::Parameter>(element::f32, shape);
    auto delta = std::make_shared<op::v0::Parameter>(element::f32, shape);
    auto bias = make_shared<op::v0::Parameter>(element::f32, Shape{shape[0]});
    auto pbroadcast = std::make_shared<op::v0::Broadcast>(bias, shape, AxisSet{1, 2, 3});
    auto conv = std::make_shared<op::v0::Convolution>(data_batch, filters);
    auto conv_bias = std::make_shared<op::v1::Add>(conv, pbroadcast);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    auto f = make_shared<Function>(conv_bias, ParameterVector{data_batch, filters, bias});

    ngraph::autodiff::Adjoints adjoints(OutputVector{conv_bias}, OutputVector{delta});

    auto d_data = adjoints.backprop_output(data_batch);
    auto d_weights = adjoints.backprop_output(filters);
    auto d_bias = adjoints.backprop_output(bias);

    auto df = make_shared<Function>(OutputVector{d_data, d_weights, d_bias},
                                    ParameterVector{data_batch, filters, bias, delta});

    pass_manager.run_passes(df);
    size_t ccg = count_ops_of_type<op::v0::ConvolutionBiasBackpropFiltersBias>(df);
    ASSERT_EQ(ccg, 1);
}

TEST(cpu_fusion, fuse_conv_relu)
{
    auto A = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto weights = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 2, 2});
    auto convolution =
        std::make_shared<op::v0::Convolution>(A, weights, Strides{1, 1}, Strides{1, 1});
    auto relu = std::make_shared<op::v0::Relu>(convolution);
    auto abs_node = std::make_shared<op::v0::Abs>(
        std::make_shared<op::v0::Abs>(std::make_shared<op::v0::Abs>(relu)));
    auto func = make_shared<Function>(abs_node, ParameterVector{A, weights});

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(pass::FusionType::REGULAR_FUSIONS);
    pass_manager.run_passes(func);
    size_t cb = count_ops_of_type<op::ConvolutionRelu>(func);
    ASSERT_GT(cb, 0);
}

// ConvolutionBiasAdd relies on an in-place fused DNNL kernel.
// Need to ensure that it is fused only when in-place buffer allocation is feasible
shared_ptr<Function> gen_conv_bias_add(bool param_input, bool result_output)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto weights = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 1});
    auto bias = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    auto conv = make_shared<op::v0::Convolution>(A, weights, Strides{1, 1}, Strides{1, 1});
    auto bias_broadcast =
        make_shared<op::v0::Broadcast>(bias, conv->get_output_shape(0), AxisSet{0, 2, 3});
    auto convbias = conv + bias_broadcast;
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto abs_B = make_shared<op::v0::Abs>(B);
    auto add = param_input ? make_shared<op::v1::Add>(convbias, B)
                           : make_shared<op::v1::Add>(convbias, abs_B);
    auto abs = make_shared<op::v0::Abs>(add);

    return result_output ? make_shared<Function>(add, ParameterVector{A, weights, bias, B})
                         : make_shared<Function>(abs, ParameterVector{A, weights, bias, B});
}

TEST(cpu_fusion, fuse_conv_bias_add)
{
    auto func_fuse = gen_conv_bias_add(false, false);
    auto func_nofuse1 = gen_conv_bias_add(true, false);
    auto func_nofuse2 = gen_conv_bias_add(false, true);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func_fuse);
    ASSERT_EQ(count_ops_of_type<op::v0::ConvolutionBiasAdd>(func_fuse), 1);

    pass_manager.run_passes(func_nofuse1);
    ASSERT_EQ(count_ops_of_type<op::v0::ConvolutionBiasAdd>(func_nofuse1), 0);

    pass_manager.run_passes(func_nofuse2);
    ASSERT_EQ(count_ops_of_type<op::v0::ConvolutionBiasAdd>(func_nofuse2), 1);
}

// ConvolutionAdd relies on an in-place fused DNNL kernel.
// Need to ensure that it is fused only when in-place buffer allocation is feasible
shared_ptr<Function> gen_conv_add(bool param_input, bool result_output)
{
    auto A = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto weights = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1, 1});
    auto conv = make_shared<op::v0::Convolution>(A, weights, Strides{1, 1}, Strides{1, 1});
    auto B = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 2});
    auto abs_B = make_shared<op::v0::Abs>(B);
    auto add =
        param_input ? make_shared<op::v1::Add>(conv, B) : make_shared<op::v1::Add>(conv, abs_B);
    auto abs = make_shared<op::v0::Abs>(add);

    return result_output ? make_shared<Function>(add, ParameterVector{A, weights, B})
                         : make_shared<Function>(abs, ParameterVector{A, weights, B});
}

TEST(cpu_fusion, fuse_conv_add)
{
    auto func_fuse = gen_conv_add(false, false);
    auto func_nofuse1 = gen_conv_add(true, false);
    auto func_nofuse2 = gen_conv_add(false, true);

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func_fuse);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionAdd>(func_fuse), 1);

    pass_manager.run_passes(func_nofuse1);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionAdd>(func_nofuse1), 0);

    pass_manager.run_passes(func_nofuse2);
    ASSERT_EQ(count_ops_of_type<op::ConvolutionAdd>(func_nofuse2), 1);
}

TEST(cpu_fusion, weight_fusion)
{
    auto param = std::make_shared<op::v0::Parameter>(element::f32, Shape{64});
    auto reshape_conv =
        std::make_shared<ngraph::op::v0::Reshape>(param, AxisVector{0}, Shape{16, 4, 1, 1});
    auto data_conv = std::make_shared<op::v0::Parameter>(element::f32, Shape{16, 4, 7, 7});
    auto tvt = &reshape_conv->get_output_tensor(0);
    auto lt_desc = std::make_shared<runtime::cpu::LayoutDescriptor>(*tvt);
    auto cvt_lt_conv = std::make_shared<runtime::cpu::op::ConvertLayout>(reshape_conv, lt_desc);
    auto conv = std::make_shared<ngraph::op::v0::Convolution>(
        data_conv, cvt_lt_conv, Strides{1, 1}, Strides{1, 1});

    auto reshape_conv_bprop =
        std::make_shared<op::v0::Reshape>(param, AxisVector{0}, Shape{16, 4, 1, 1});
    auto dummy_arg_conv_bprop =
        std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 16, 7, 7});
    auto tvt_bprop = &reshape_conv_bprop->get_output_tensor(0);
    auto lt_desc_bprop = std::make_shared<runtime::cpu::LayoutDescriptor>(*tvt_bprop);
    auto cvt_lt_conv_bprop =
        std::make_shared<runtime::cpu::op::ConvertLayout>(reshape_conv_bprop, lt_desc_bprop);
    auto conv_bprop = std::make_shared<op::v0::ConvolutionBackpropData>(Shape{1, 4, 7, 7},
                                                                        cvt_lt_conv_bprop,
                                                                        dummy_arg_conv_bprop,
                                                                        Strides{1, 1},
                                                                        Strides{1, 1},
                                                                        CoordinateDiff{0, 0},
                                                                        CoordinateDiff{0, 0},
                                                                        Strides{1, 1});

    auto conv_relu = std::make_shared<op::v0::Relu>(conv);
    auto conv_bprop_abs = std::make_shared<op::v0::Abs>(conv_bprop);

    auto f = make_shared<Function>(OutputVector{conv_relu, conv_bprop_abs},
                                   ParameterVector{param, data_conv, dummy_arg_conv_bprop});

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUPostLayoutOptimizations>();
    pass_manager.run_passes(f);

    auto new_conv_bprop_data = conv_bprop_abs->get_argument(0);
    auto new_convert_layout = new_conv_bprop_data->get_argument(0);

    ASSERT_EQ(as_type_ptr<runtime::cpu::op::ConvertLayout>(new_convert_layout->get_argument(0)),
              cvt_lt_conv);
}

TEST(cpu_fusion, max_pool_with_indices)
{
    Shape shape_a{10, 3, 28, 28};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape window_shape{2, 2};
    auto max_pool = std::make_shared<op::v0::MaxPool>(input, window_shape);
    auto C = std::make_shared<op::v0::Parameter>(element::f32, max_pool->get_output_shape(0));

    ngraph::autodiff::Adjoints adjoints(ngraph::OutputVector{max_pool}, ngraph::OutputVector{C});

    auto dinput = adjoints.backprop_output(input);

    auto df = std::make_shared<Function>(OutputVector{dinput}, ParameterVector{input, C});

    auto f = std::make_shared<Function>(OutputVector{max_pool}, ParameterVector{input});

    {
        OutputVector nv_cwi;
        pass::Manager pass_manager;
        pass_manager.register_pass<runtime::cpu::pass::CPUWorkspaceInsertion>(nv_cwi);
        pass_manager.run_passes(df);
    }

    size_t index = f->get_results().at(0)->input(0).get_source_output().get_index();
    EXPECT_EQ(index, 0);

    auto maxpool_with_indices = df->get_results().at(0)->get_argument(0);
    index = maxpool_with_indices->input(2).get_source_output().get_index();
    EXPECT_EQ(index, 1);
}

static std::shared_ptr<ngraph::Function> make_forward_function()
{
    Shape shape_a{10, 3, 28, 28};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, shape_a);
    Shape window_shape{2, 2};
    auto max_pool = std::make_shared<op::v0::MaxPool>(input, window_shape);
    auto neg = std::make_shared<op::v0::Negative>(max_pool);
    auto absn = std::make_shared<op::v0::Abs>(max_pool);
    return std::make_shared<Function>(OutputVector{max_pool, neg, absn}, ParameterVector{input});
}

static std::pair<std::shared_ptr<ngraph::Function>, OutputVector>
    make_backward_function(std::shared_ptr<ngraph::Function> f)
{
    // get parameters
    ParameterVector back_parameters = f->get_parameters();

    ngraph::OutputVector adjoints;
    ngraph::OutputVector outputs;
    for (auto Y : f->get_results())
    {
        // Get the output
        // Create the Adjoint
        auto C = std::make_shared<ngraph::op::v0::Parameter>(Y->get_output_element_type(0),
                                                             Y->get_output_shape(0));
        outputs.push_back(Y);
        adjoints.push_back(C);
    }

    ngraph::autodiff::Adjoints adjoint{outputs, adjoints};

    // Perform autodiff
    OutputVector dYdXs(back_parameters.size());
    transform(back_parameters.begin(),
              back_parameters.end(),
              dYdXs.begin(),
              [&adjoint](const std::shared_ptr<Node>& X) { return adjoint.backprop_output(X); });

    // create the backward function
    ParameterVector param_adjoints;
    for (auto n : adjoints)
        param_adjoints.push_back(as_type_ptr<ngraph::op::v0::Parameter>(n.get_node_shared_ptr()));
    back_parameters.insert(back_parameters.begin(), param_adjoints.begin(), param_adjoints.end());

    return {std::make_shared<ngraph::Function>(dYdXs, back_parameters), adjoints};
}

void optimize_graph(std::shared_ptr<ngraph::Function>& f, std::shared_ptr<ngraph::Function> bf)
{
    // start by removing excess reshapes
    OutputVector nv_cwi;
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<runtime::cpu::pass::CPUWorkspaceInsertion>(nv_cwi);

    pass_manager.run_passes(f);
    pass_manager.run_passes(bf);
    if (nv_cwi.size() > 0)
    {
        OutputVector new_outputs;
        for (auto r : f->get_results())
        {
            new_outputs.push_back(r->get_argument(0));
        }

        new_outputs.insert(new_outputs.end(), nv_cwi.begin(), nv_cwi.end());
        f = std::make_shared<ngraph::Function>(new_outputs, f->get_parameters());
    }

    ngraph::NodeVector dYdXs;
    for (size_t i = 0; i < bf->get_output_size(); ++i)
    {
        dYdXs.push_back(bf->get_output_op(i)->get_argument(0));
    }

    ngraph::OutputVector combined_outputs;
    for (auto r : f->get_results())
    {
        combined_outputs.push_back(r);
    }

    combined_outputs.insert(combined_outputs.end(), dYdXs.begin(), dYdXs.end());

    ParameterVector combined_parameters = f->get_parameters();
    ParameterVector back_parameters = bf->get_parameters();

    combined_parameters.insert(
        combined_parameters.end(), back_parameters.begin(), back_parameters.end());
    auto combinedf = std::make_shared<ngraph::Function>(combined_outputs, combined_parameters);
    // rerun Reshape elimination to help simplify the graph again, run CPUFusion
    // this replaces nodes in both f and bf due to shared-ptr - ness
    ngraph::pass::Manager pass_manager_comb;
    pass_manager_comb.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager_comb.register_pass<ngraph::runtime::cpu::pass::CPUFusion>();
    pass_manager_comb.run_passes(combinedf);
}

TEST(cpu_fusion, maxpool_with_indices_in_mxnet)
{
    auto f = make_forward_function();
    auto bfa = make_backward_function(f);
    auto maybe_bf = bfa.first;
    auto adjoints = bfa.second;
    optimize_graph(f, maybe_bf);
    auto fprop_cache = ngraph::cache_fprop(f, maybe_bf);

    auto mpwi_bprop = fprop_cache.bprop->get_results().at(0)->get_argument(0);
    ASSERT_TRUE(as_type_ptr<op::v0::Parameter>(mpwi_bprop->get_argument(0)));
    ASSERT_TRUE(as_type_ptr<op::v0::Parameter>(mpwi_bprop->get_argument(2)));
}

static std::shared_ptr<Function>
    create_rnn_input_linear_transformation_function(size_t num_timesteps, bool data_is_4d = false)
{
    auto W = std::make_shared<op::v0::Parameter>(element::f32, Shape{400, 50});
    auto bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{400});
    ParameterVector params{W, bias};
    auto create_graph = [&]() -> std::shared_ptr<Node> {
        auto data_param =
            (data_is_4d) ? std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 5, 1, 50})
                         : std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 1, 50});
        params.push_back(data_param);
        auto reshape_axis_order = data_is_4d ? AxisVector{0, 1, 2, 3} : AxisVector{0, 1, 2};
        auto data_param_reshape =
            std::make_shared<op::v0::Reshape>(data_param, reshape_axis_order, Shape{10, 50});
        auto W_reshape = std::make_shared<op::v0::Reshape>(W, AxisVector{1, 0}, Shape{50, 400});
        auto dot = std::make_shared<op::v0::Dot>(data_param_reshape, W_reshape);
        auto bias_broadcast =
            make_shared<op::v0::Broadcast>(bias, dot->get_output_shape(0), AxisSet{0});
        auto add_bias = std::make_shared<op::v1::Add>(dot, bias_broadcast);
        return move(add_bias);
    };

    NodeVector graph_nodes;
    for (size_t i = 0; i < num_timesteps; i++)
    {
        graph_nodes.push_back(create_graph());
    }
    auto concat = std::make_shared<op::v0::Concat>(graph_nodes, 0);
    return make_shared<Function>(OutputVector{concat}, params);
}

TEST(cpu_fusion, fuse_rnn_input_across_time_steps)
{
    auto func = create_rnn_input_linear_transformation_function(10);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPURnnMatFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func);
    size_t ref_matmulbias_count = 1;
    auto matmulbias_count = count_ops_of_type<op::MatmulBias>(func);
    EXPECT_EQ(ref_matmulbias_count, matmulbias_count);
}

TEST(cpu_fusion, fuse_rnn_input_across_time_steps_4d_data)
{
    auto func = create_rnn_input_linear_transformation_function(10, true);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPURnnMatFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.run_passes(func);
    size_t ref_matmulbias_count = 10; // no CPURnnMatFusion transformations
    auto matmulbias_count = count_ops_of_type<op::MatmulBias>(func);
    EXPECT_EQ(ref_matmulbias_count, matmulbias_count);
}

#ifndef NGRAPH_JSON_DISABLE
// Tests that rely on deserializing json files
TEST(cpu_fusion, fuse_conv_bias)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<ngraph::runtime::cpu::pass::CPUFusion>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "conv_bias.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t cb = count_ops_of_type<op::v0::ConvolutionBias>(func);
    ASSERT_GT(cb, 0);
}

TEST(cpu_fusion, gemm_mlp)
{
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/mnist_mlp_forward.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(pass::FusionType::REGULAR_FUSIONS);
    pass_manager.run_passes(func);
    auto mmbs = count_ops_of_type<op::MatmulBias>(func);
    ASSERT_EQ(mmbs, 3);
}

TEST(cpu_fusion, fuse_fprop_bn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(pass::FusionType::REGULAR_FUSIONS);
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/bn_fprop_b2c3h2w2.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::v0::BatchNormTraining>(func);
    ASSERT_EQ(ccg, 1);
}

TEST(cpu_fusion, sigmoid_multiply_fusion)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::CoreFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/3_lstm_cell_forward.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t ccg = count_ops_of_type<op::SigmoidMultiply>(func);
    ASSERT_EQ(ccg, 18);
}

TEST(cpu_fusion, fuse_bi_directional_rnn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::cpu::pass::RNNFusion>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    pass_manager.register_pass<runtime::cpu::pass::MultiLayerRNNFusion>();
    pass_manager.register_pass<runtime::cpu::pass::BiDirectionalRnn>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/lstm_bi_directional.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    // Bidirectional graph pass will folds the reverse seq
    auto rev_seq_ops = get_ops_of_type<op::v0::Reverse>(func);
    auto rnn_ops = get_ops_of_type<op::Rnn>(func);
    EXPECT_EQ(rev_seq_ops.size(), 0);
    // fuse two bi-directional rnn layers in to one DNNL Op
    EXPECT_EQ(rnn_ops.size(), 1);
}

TEST(cpu_fusion, rnn_fusion_from_json_model)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::CPURnnMatFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>(pass::FusionType::REGULAR_FUSIONS);
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/rnn-10-step-fusion-test.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    const size_t NUM_STEPS = 10;
    auto mmb_predicate = [=](std::shared_ptr<Node> node) {
        auto users = node->get_users();
        return (users.size() == NUM_STEPS) &&
               std::all_of(begin(users), end(users), [](std::shared_ptr<Node> n) {
                   return as_type_ptr<op::v0::Slice>(n) != nullptr;
               });
    };

    auto mmbs = get_ops_of_type<op::MatmulBias>(func);
    ASSERT_TRUE(std::any_of(begin(mmbs), end(mmbs), mmb_predicate));
}

TEST(cpu_fusion, fuse_lstm_cells)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/2rnn_layer_3lstm_cell.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    auto lstm_ops = get_ops_of_type<op::Lstm>(func);
    EXPECT_EQ(lstm_ops.size(), 6);
}

TEST(cpu_fusion, fuse_2_layer_rnn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::cpu::pass::RNNFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/2rnn_layer_3lstm_cell.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t count = count_ops_of_type<op::Rnn>(func);
    auto rnn_ops = get_ops_of_type<op::Rnn>(func);
    EXPECT_EQ(rnn_ops.size(), count);
    for (auto& node : rnn_ops)
    {
        EXPECT_EQ(node->get_num_timesteps(), node->get_src_sequence_length());
    }
}

TEST(cpu_fusion, fuse_1_layer_rnn)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::cpu::pass::RNNFusion>();
    const string json_path =
        file_util::path_join(SERIALIZED_ZOO, "mxnet/1rnn_layer_3lstm_cell.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    pass_manager.run_passes(func);
    size_t count = count_ops_of_type<op::Rnn>(func);
    auto rnn_ops = get_ops_of_type<op::Rnn>(func);
    EXPECT_EQ(rnn_ops.size(), 1);
    EXPECT_EQ(rnn_ops.size(), count);
    for (auto& node : rnn_ops)
    {
        EXPECT_EQ(node->get_num_timesteps(), node->get_src_sequence_length());
    }
}

#endif
