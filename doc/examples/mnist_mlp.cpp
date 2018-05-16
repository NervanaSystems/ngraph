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

#include <cstdio>
#include <functional>
#include <iostream>
#include <list>
#include <math.h>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

#include <ngraph/autodiff/adjoints.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/ngraph.hpp>

#include "mnist_loader.hpp"
#include "tensor_utils.hpp"

using namespace ngraph;

size_t
    accuracy_count(const std::shared_ptr<runtime::TensorView>& t_softmax,
                   const std::shared_ptr<runtime::TensorView>& t_Y)
{
    const Shape& softmax_shape = t_softmax->get_shape();
    size_t batch_size = softmax_shape.at(0);
    size_t label_count = softmax_shape.at(1);
    const Shape& Y_shape = t_Y->get_shape();
    if (Y_shape.size() != 1 || Y_shape.at(0) != batch_size)
    {
        throw std::invalid_argument(
            "Y and softmax shapes are incompatible");
    }
    size_t softmax_pos = 0;
    size_t count = 0;
    for (size_t i = 0; i < batch_size; ++i)
    {
        float max_value = get_scalar<float>(t_softmax, softmax_pos++);
        size_t max_idx = 0;
        for (size_t j = 1; j < label_count; ++j)
        {
            float value = get_scalar<float>(t_softmax, softmax_pos++);
            if (value > max_value)
            {
                max_value = value;
                max_idx = j;
            }
        }
        float correct_idx = get_scalar<float>(t_Y, i);
        if (static_cast<size_t>(correct_idx) ==
            static_cast<size_t>(max_idx))
        {
            count++;
        }
    }
    return count;
}

float test_accuracy(MNistDataLoader& loader,
                    std::shared_ptr<runtime::Backend> backend,
                    std::shared_ptr<Function> function,
                    const std::shared_ptr<runtime::TensorView>& t_X,
                    const std::shared_ptr<runtime::TensorView>& t_Y,
                    const std::shared_ptr<runtime::TensorView>& t_softmax,
                    const std::shared_ptr<runtime::TensorView>& t_W0,
                    const std::shared_ptr<runtime::TensorView>& t_b0,
                    const std::shared_ptr<runtime::TensorView>& t_W1,
                    const std::shared_ptr<runtime::TensorView>& t_b1)
{
    loader.reset();
    size_t batch_size = loader.get_batch_size();
    size_t acc_count = 0;
    size_t sample_count = 0;
    while (loader.get_epoch() < 1)
    {
        loader.load();
        t_X->write(loader.get_image_floats(),
                   0,
                   loader.get_image_batch_size() * sizeof(float));
        t_Y->write(loader.get_label_floats(),
                   0,
                   loader.get_label_batch_size() * sizeof(float));
        backend->call(
            function, {t_softmax}, {t_X, t_W0, t_b0, t_W1, t_b1});
        size_t acc = accuracy_count(t_softmax, t_Y);
        acc_count += acc;
        sample_count += batch_size;
    }
    return static_cast<float>(acc_count) /
           static_cast<float>(sample_count);
}

int main(int argc, const char* argv[])
{
    size_t epochs = 5;
    size_t batch_size = 128;
    size_t output_size = 10;

    size_t l0_size = 600;
    size_t l1_size = output_size;
    float log_min = static_cast<float>(exp(-50.0));
    MNistDataLoader test_loader{
        batch_size, MNistImageLoader::TEST, MNistLabelLoader::TEST};
    MNistDataLoader train_loader{
        batch_size, MNistImageLoader::TRAIN, MNistLabelLoader::TRAIN};
    train_loader.open();
    test_loader.open();
    size_t input_size =
        train_loader.get_columns() * train_loader.get_rows();

    // The data input
    auto X = std::make_shared<op::Parameter>(
        element::f32, Shape{batch_size, input_size});

    // Layer 0
    auto W0 = std::make_shared<op::Parameter>(element::f32,
                                              Shape{input_size, l0_size});
    auto b0 =
        std::make_shared<op::Parameter>(element::f32, Shape{l0_size});
    auto l0_dot = std::make_shared<op::Dot>(X, W0, 1);
    auto b0_broadcast = std::make_shared<op::Broadcast>(
        b0, Shape{batch_size, l0_size}, AxisSet{0});
    auto l0 = std::make_shared<op::Relu>(l0_dot + b0_broadcast);

    // Layer 1
    auto W1 = std::make_shared<op::Parameter>(element::f32,
                                              Shape{l0_size, l1_size});
    auto b1 =
        std::make_shared<op::Parameter>(element::f32, Shape{l1_size});
    auto l1_dot = std::make_shared<op::Dot>(l0, W1, 1);
    auto b1_broadcast = std::make_shared<op::Broadcast>(
        b1, Shape{batch_size, l1_size}, AxisSet{0});
    auto l1 = l1_dot + b1_broadcast;

    // Softmax
    auto softmax = std::make_shared<op::Softmax>(l1, AxisSet{1});

    // Loss computation
    auto Y =
        std::make_shared<op::Parameter>(element::f32, Shape{batch_size});
    auto labels =
        std::make_shared<op::OneHot>(Y, Shape{batch_size, output_size}, 1);
    auto softmax_clip_value = std::make_shared<op::Constant>(
        element::f32, Shape{}, std::vector<float>{log_min});
    auto softmax_clip_broadcast = std::make_shared<op::Broadcast>(
        softmax_clip_value, Shape{batch_size, output_size}, AxisSet{0, 1});
    auto softmax_clip =
        std::make_shared<op::Maximum>(softmax, softmax_clip_broadcast);
    auto softmax_log = std::make_shared<op::Log>(softmax_clip);
    auto prod = std::make_shared<op::Multiply>(softmax_log, labels);
    auto N = std::make_shared<op::Parameter>(element::f32, Shape{});
    auto loss = std::make_shared<op::Divide>(
        std::make_shared<op::Sum>(prod, AxisSet{0, 1}), N);

    // Backprop
    // Each of W0, b0, W1, and b1
    auto learning_rate =
        std::make_shared<op::Parameter>(element::f32, Shape{});
    auto delta = -learning_rate * loss;

    // Updates
    ngraph::autodiff::Adjoints adjoints(NodeVector{loss},
                                        NodeVector{delta});
    auto W0_next = W0 + adjoints.backprop_node(W0);
    auto b0_next = b0 + adjoints.backprop_node(b0);
    auto W1_next = W1 + adjoints.backprop_node(W1);
    auto b1_next = b1 + adjoints.backprop_node(b1);

    // Get the backend
    auto backend = runtime::Backend::create("CPU");

    // Allocate and randomly initialize variables
    auto t_W0 = make_output_tensor(backend, W0, 0);
    auto t_b0 = make_output_tensor(backend, b0, 0);
    auto t_W1 = make_output_tensor(backend, W1, 0);
    auto t_b1 = make_output_tensor(backend, b1, 0);

    std::function<float()> rand(
        std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                  std::default_random_engine(0)));
    randomize(rand, t_W0);
    randomize(rand, t_b0);
    randomize(rand, t_W1);
    randomize(rand, t_b1);

    // Allocate inputs
    auto t_X = make_output_tensor(backend, X, 0);
    auto t_Y = make_output_tensor(backend, Y, 0);

    auto t_learning_rate = make_output_tensor(backend, learning_rate, 0);
    auto t_N = make_output_tensor(backend, N, 0);
    set_scalar(t_N, static_cast<float>(batch_size), 0);

    // Allocate updated variables
    auto t_W0_next = make_output_tensor(backend, W0_next, 0);
    auto t_b0_next = make_output_tensor(backend, b0_next, 0);
    auto t_W1_next = make_output_tensor(backend, W1_next, 0);
    auto t_b1_next = make_output_tensor(backend, b1_next, 0);

    auto t_loss = make_output_tensor(backend, loss, 0);
    auto t_softmax = make_output_tensor(backend, softmax, 0);

    // Train
    // X, Y, learning_rate, W0, b0, W1, b1 -> loss, softmax, W0_next, b0_next, W1_next, b1_next
    NodeMap train_node_map;
    auto train_function = clone_function(
        Function(
            NodeVector{loss, softmax, W0_next, b0_next, W1_next, b1_next},
            op::ParameterVector{X, Y, N, learning_rate, W0, b0, W1, b1}),
        train_node_map);

    // Plain inference
    // X, W0, b0, W1, b1 -> softmax
    NodeMap inference_node_map;
    auto inference_function =
        clone_function(Function(NodeVector{softmax},
                                op::ParameterVector{X, W0, b0, W1, b1}),
                       inference_node_map);

    set_scalar(t_learning_rate, .03f);

    size_t last_epoch = 0;
    while (train_loader.get_epoch() < epochs)
    {
        train_loader.load();
        t_X->write(train_loader.get_image_floats(),
                   0,
                   train_loader.get_image_batch_size() * sizeof(float));
        t_Y->write(train_loader.get_label_floats(),
                   0,
                   train_loader.get_label_batch_size() * sizeof(float));
        backend->call(
            train_function,
            {t_loss,
             t_softmax,
             t_W0_next,
             t_b0_next,
             t_W1_next,
             t_b1_next},
            {t_X, t_Y, t_N, t_learning_rate, t_W0, t_b0, t_W1, t_b1});

        t_W0.swap(t_W0_next);
        t_b0.swap(t_b0_next);
        t_W1.swap(t_W1_next);
        t_b1.swap(t_b1_next);

        if (train_loader.get_epoch() != last_epoch)
        {
            last_epoch = train_loader.get_epoch();
            std::cout << "Test accuracy: "
                      << test_accuracy(test_loader,
                                       backend,
                                       inference_function,
                                       t_X,
                                       t_Y,
                                       t_softmax,
                                       t_W0,
                                       t_b0,
                                       t_W1,
                                       t_b1)
                      << std::endl;
        }
    }

    return 0;
}
