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

#include <iomanip>

#include "benchmark.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "random.hpp"

using namespace std;
using namespace ngraph;

multimap<size_t, string>
    aggregate_timing_details(const vector<runtime::PerformanceCounter>& perf_data,
                             shared_ptr<Function> func)
{
    unordered_map<string, shared_ptr<Node>> node_map;
    vector<shared_ptr<Function>> fs;
    traverse_functions(func, [&](shared_ptr<Function> f) { fs.push_back(f); });
    for (shared_ptr<Function> f : fs)
    {
        for (shared_ptr<Node> node : f->get_ops())
        {
            node_map.insert({node->get_name(), node});
        }
    }

    unordered_map<string, size_t> timing;
    for (const runtime::PerformanceCounter& p : perf_data)
    {
        shared_ptr<Node> node = node_map.at(p.name());
        string op = p.name().substr(0, p.name().find('_'));
        string shape_name = "{" + join(node->get_outputs()[0].get_shape()) + "}";
        timing[op + shape_name] += p.microseconds();
    }

    multimap<size_t, string> rc;
    for (const pair<string, size_t>& t : timing)
    {
        rc.insert({t.second, t.first});
    }
    return rc;
}

multimap<size_t, string> aggregate_timing(const vector<runtime::PerformanceCounter>& perf_data)
{
    unordered_map<string, size_t> timing;
    for (const runtime::PerformanceCounter& p : perf_data)
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

void run_benchmark(const string& json_path,
                   const string& backend_name,
                   size_t iterations,
                   bool timing_detail)
{
    stopwatch timer;
    timer.start();
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> f = deserialize(ss);
    timer.stop();
    cout << "deserialize time: " << timer.get_milliseconds() << "ms" << endl;
    run_benchmark(f, backend_name, iterations, timing_detail);
}

void print_times(const multimap<size_t, string>& timing)
{
    // set the column widths
    int name_width = 0;
    int time_width = 0;
    for (const pair<size_t, string>& p : timing)
    {
        name_width = max(name_width, static_cast<int>(p.second.size()));
        stringstream ss;
        ss.imbue(locale(""));
        ss << p.first;
        time_width = max(time_width, static_cast<int>(ss.str().size()));
    }
    for (auto it = timing.rbegin(); it != timing.rend(); it++)
    {
        cout << setw(name_width + 2) << left << it->second << " " << setw(time_width + 2) << right
             << it->first << "us\n";
    }
}

static default_random_engine s_random_engine;

template <typename T>
void init_int_tv(shared_ptr<runtime::TensorView> tv, T min, T max)
{
    uniform_int_distribution<T> dist(min, max);
    std::vector<T> vec = read_vector<T>(tv);
    for (T& element : vec)
    {
        element = dist(s_random_engine);
    }
    write_vector(tv, vec);
}

template <typename T>
void init_real_tv(shared_ptr<runtime::TensorView> tv, T min, T max)
{
    uniform_real_distribution<T> dist(min, max);
    std::vector<T> vec = read_vector<T>(tv);
    for (T& element : vec)
    {
        element = dist(s_random_engine);
    }
    write_vector(tv, vec);
}

static void random_init(shared_ptr<runtime::TensorView> tv)
{
    element::Type et = tv->get_tensor().get_element_type();
    if (et == element::boolean)
    {
        init_int_tv<char>(tv, 0, 1);
    }
    else if (et == element::f32)
    {
        init_real_tv<float>(tv, -1, 1);
    }
    else if (et == element::f64)
    {
        init_real_tv<double>(tv, -1, 1);
    }
    else if (et == element::i8)
    {
        init_int_tv<int8_t>(tv, -1, 1);
    }
    else if (et == element::i16)
    {
        init_int_tv<int16_t>(tv, -1, 1);
    }
    else if (et == element::i32)
    {
        init_int_tv<int32_t>(tv, -1, 1);
    }
    else if (et == element::i64)
    {
        init_int_tv<int64_t>(tv, -1, 1);
    }
    else if (et == element::u8)
    {
        init_int_tv<uint8_t>(tv, 0, 1);
    }
    else if (et == element::u16)
    {
        init_int_tv<uint16_t>(tv, 0, 1);
    }
    else if (et == element::u32)
    {
        init_int_tv<uint32_t>(tv, 0, 1);
    }
    else if (et == element::u64)
    {
        init_int_tv<uint64_t>(tv, 0, 1);
    }
    else
    {
        throw runtime_error("unsupported type");
    }
}

void run_benchmark(shared_ptr<Function> f,
                   const string& backend_name,
                   size_t iterations,
                   bool timing_detail)
{
    stopwatch timer;
    timer.start();
    auto backend = runtime::Backend::create(backend_name);
    backend->enable_performance_data(f, timing_detail);
    backend->compile(f);
    timer.stop();
    cout.imbue(locale(""));
    cout << "compile time: " << timer.get_milliseconds() << "ms" << endl;

    vector<shared_ptr<runtime::TensorView>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto tensor = backend->create_tensor(param->get_element_type(), param->get_shape());
        random_init(tensor);
        args.push_back(tensor);
    }
    vector<shared_ptr<runtime::TensorView>> results;
    for (shared_ptr<Node> out : f->get_results())
    {
        auto result = backend->create_tensor(out->get_element_type(), out->get_shape());
        results.push_back(result);
    }

    stopwatch t1;
    t1.start();
    for (size_t i = 0; i < static_cast<size_t>(iterations); i++)
    {
        backend->call(f, results, args);
    }
    t1.stop();
    float time = t1.get_milliseconds();
    cout << time / iterations << "ms per iteration" << endl;

    vector<runtime::PerformanceCounter> perf_data = backend->get_performance_data(f);
    sort(perf_data.begin(),
         perf_data.end(),
         [](const runtime::PerformanceCounter& p1, const runtime::PerformanceCounter& p2) {
             return p1.total_microseconds() > p2.total_microseconds();
         });
    multimap<size_t, string> timing = aggregate_timing(perf_data);
    multimap<size_t, string> timing_details = aggregate_timing_details(perf_data, f);

    cout << "\n---- Aggregate times per op type ----\n";
    print_times(timing);

    cout << "\n---- Aggregate times per op type/shape ----\n";
    print_times(timing_details);
}
