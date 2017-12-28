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
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/random.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

static multimap<size_t, string>
    agregate_timing(const vector<runtime::cpu::PerformanceCounter>& perf_data)
{
    unordered_map<string, size_t> timing;
    for (const runtime::cpu::PerformanceCounter& p : perf_data)
    {
        string op = p.name().substr(0, p.name().find('_'));
        timing[op] += p.microseconds();
    }

    multimap<size_t, string> rc;
    for (const pair<string, size_t>& t : timing)
    {
        rc.insert({t.second, t.first});
    }
    return rc;
}

// Starting point CPU: 1.2ms/iteration

shared_ptr<runtime::TensorView> make_tensor(runtime::Backend& backend, shared_ptr<Node> value)
{
    shared_ptr<runtime::TensorView> arg =
        backend.make_primary_tensor_view(value->get_element_type(), value->get_shape());
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
        auto arg = make_tensor(*backend, param);
        rng.initialize(arg);
        args.push_back(arg);
    }
    auto result = make_tensor(*backend, f->get_output_op(0));

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
    bool emit_timing = (std::getenv("NGRAPH_CPU_EMIT_TIMING") != nullptr);
    if (!emit_timing)
    {
        cout << "To get per-op timing set the environment variable NGRAPH_CPU_EMIT_TIMING\n";
    }

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
        auto arg = make_tensor(*backend, param);
        rng.initialize(arg);
        args.push_back(arg);
    }
    auto result = make_tensor(*backend, f->get_output_op(0));

    stopwatch t1;
    t1.start();
    float count = 30;
    for (size_t i = 0; i < static_cast<size_t>(count); i++)
    {
        cf->call(args, {result});
    }
    t1.stop();
    float time = t1.get_milliseconds();
    cout << time / count << "ms per iteration\n";

    if (emit_timing)
    {
        vector<runtime::cpu::PerformanceCounter> perf_data = cpu_cf->get_performance_data();
        sort(perf_data.begin(),
             perf_data.end(),
             [](const runtime::cpu::PerformanceCounter& p1,
                const runtime::cpu::PerformanceCounter& p2) {
                 return p1.total_microseconds() > p2.total_microseconds();
             });
        multimap<size_t, string> timing = agregate_timing(perf_data);
        for (auto it = timing.rbegin(); it != timing.rend(); it++)
        {
            cout.imbue(locale(""));
            cout << setw(15) << left << it->second << " " << setw(10) << right << it->first
                 << "us\n";
        }
        // size_t count = 30;
        // for (const runtime::cpu::PerformanceCounter& p : perf_data)
        // {
        //     if (--count == 0)
        //     {
        //         break;
        //     }
        //     cout.imbue(locale(""));
        //     cout << setw(15) << left << p.name() << " " << setw(10) << right
        //          << p.microseconds() << "us\n";
        // }
    }
}

//
// Benchmarks a graph that concatenates six 32x1x200 arrays along the middle axis.
//
TEST(benchmark, concat_32x1x200_axis1_6)
{
    const size_t n_arrays = 6;
    Shape shape_of_each_array = Shape{32, 1, 200};
    size_t concatenation_axis = 1;

    Shape result_shape;
    result_shape = shape_of_each_array;
    result_shape[concatenation_axis] *= n_arrays;

    size_t elements_per_array = 1;
    for (size_t d : shape_of_each_array)
    {
        elements_per_array *= d;
    }

    vector<vector<float>> data_arrays(n_arrays);
    for (size_t i = 0; i < n_arrays; i++)
    {
        data_arrays[i] = vector<float>(elements_per_array);
        for (size_t j = 0; j < elements_per_array; j++)
        {
            data_arrays[i][j] = float(j + 1);
        }
    }

    bool using_ref_kernels = (std::getenv("NGRAPH_CPU_USE_REF_KERNELS") != nullptr);

    vector<std::string> backend_names{"INTERPRETER", "CPU"};
    vector<int> n_runs{200, 200, using_ref_kernels ? 200 : 200000}; // one for each backend
    vector<std::function<void()>> test_callbacks;                   // one for each backend
    vector<std::shared_ptr<runtime::TensorView>> result_tvs;        // one for each backend

    for (std::string backend_name : backend_names)
    {
        vector<std::shared_ptr<op::Parameter>> params(n_arrays);
        vector<std::shared_ptr<Node>> params_as_nodes(n_arrays);
        for (size_t i = 0; i < n_arrays; i++)
        {
            auto param = make_shared<op::Parameter>(element::f32, shape_of_each_array);
            params[i] = param;
            params_as_nodes[i] = param;
        }

        auto concat = make_shared<op::Concat>(params_as_nodes, concatenation_axis);
        auto f = make_shared<Function>(concat, params);

        auto manager = runtime::Manager::get(backend_name);
        auto external = manager->compile(f);
        auto backend = manager->allocate_backend();
        auto cf = backend->make_call_frame(external);

        vector<shared_ptr<runtime::Value>> input_vals;

        for (size_t i = 0; i < n_arrays; i++)
        {
            auto tv = backend->make_primary_tensor_view(element::f32, shape_of_each_array);
            copy_data(tv, data_arrays[i]);
            input_vals.push_back(tv);
        }

        auto result_tv = backend->make_primary_tensor_view(element::f32, result_shape);
        result_tvs.push_back(result_tv);

        std::function<void()> cb = [input_vals, result_tv, cf]() {
            cf->call(input_vals, {result_tv});
        };

        test_callbacks.push_back(cb);
    }

    for (size_t i = 0; i < backend_names.size(); i++)
    {
        std::cout << backend_names[i] << ": " << n_runs[i] << " tests in " << std::flush;

        stopwatch sw;
        std::function<void()> cb = test_callbacks[i];

        sw.start();
        for (int j = 0; j < n_runs[i]; j++)
        {
            cb();
        }
        sw.stop();

        std::cout << sw.get_milliseconds() << "ms (" << (sw.get_microseconds() / n_runs[i])
                  << " us/test)" << std::endl;
    }

    for (size_t i = 1; i < backend_names.size(); i++)
    {
        std::cout << "Verifying " << backend_names[i] << " result against " << backend_names[0]
                  << "..." << std::flush;

        if (result_tvs[i]->get_vector<float>() == result_tvs[0]->get_vector<float>())
        {
            std::cout << " OK" << std::endl;
        }
        else
        {
            std::cout << " FAILED" << std::endl;
            ADD_FAILURE();
        }
    }
}
