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

#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/random.hpp"

using namespace std;
using namespace ngraph;

// Starting point CPU: 1.2ms/iteration

shared_ptr<runtime::TensorView> make_tensor(runtime::Backend& backend, const ValueType& value)
{
    shared_ptr<runtime::TensorView> arg =
        backend.make_primary_tensor_view(value.get_element_type(), value.get_shape());
    return arg;
}

TEST(benchmark, mxnet_mnist_mlp_forward)
{
    test::Uniform<float> rng{-1, 1, 0};
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/mnist_mlp_forward.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> f = ngraph::deserialize(ss);

    auto manager = runtime::Manager::get("CPU");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    vector<shared_ptr<runtime::Value>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto arg = make_tensor(*backend, *(param->get_value_type()));
        rng.initialize(arg);
        args.push_back(arg);
    }
    shared_ptr<const ValueType> result_type = f->get_result_type();
    auto result = make_tensor(*backend, *result_type);

    stopwatch t1;
    t1.start();
    float count = 1000;
    for (size_t i = 0; i < static_cast<size_t>(count); i++)
    {
        cf->call(args, {result});
    }
    t1.stop();
    float time = t1.get_milliseconds();
    cout << time / count << "ms per iteration\n";
}

TEST(benchmark, mxnet_10_bucket_lstm)
{
    test::Uniform<float> rng{-1, 1, 0};
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/10_bucket_LSTM.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> f = ngraph::deserialize(ss);

    auto manager = runtime::Manager::get("CPU");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    runtime::cpu::CPU_CallFrame* cpu_cf = static_cast<runtime::cpu::CPU_CallFrame*>(cf.get());

    vector<shared_ptr<runtime::Value>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto arg = make_tensor(*backend, *(param->get_value_type()));
        rng.initialize(arg);
        args.push_back(arg);
    }
    shared_ptr<const ValueType> result_type = f->get_result_type();
    auto result = make_tensor(*backend, *result_type);

    stopwatch t1;
    t1.start();
    float count = 10;
    for (size_t i = 0; i < static_cast<size_t>(count); i++)
    {
        cf->call(args, {result});
    }
    t1.stop();
    float time = t1.get_milliseconds();
    cout << time / count << "ms per iteration\n";

    vector<runtime::cpu::PerformanceCounter> perf_data = cpu_cf->get_performance_data();
    sort(
        perf_data.begin(),
        perf_data.end(),
        [](const runtime::cpu::PerformanceCounter& p1, const runtime::cpu::PerformanceCounter& p2) {
            return p1.total_microseconds() > p2.total_microseconds();
        });
    for (const runtime::cpu::PerformanceCounter& p : perf_data)
    {
        NGRAPH_INFO << p.name() << ", " << p.total_microseconds();
    }
}
