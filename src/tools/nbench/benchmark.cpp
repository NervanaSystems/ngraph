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
#include <random>

#include "benchmark.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

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
    unordered_map<string, size_t> count;
    for (const runtime::PerformanceCounter& p : perf_data)
    {
        shared_ptr<Node> node = node_map.at(p.name());
        string op = p.name().substr(0, p.name().find('_'));
        string shape_name = " {" + join(node->get_outputs()[0].get_shape()) + "} ";
        timing[op + shape_name] += p.microseconds();
        count[op + shape_name] += 1;
    }

    multimap<size_t, string> rc;
    for (const pair<string, size_t>& t : timing)
    {
        rc.insert({t.second, t.first + to_string(count[t.first])});
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
    size_t size = tv->get_element_count();
    uniform_int_distribution<T> dist(min, max);
    vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(s_random_engine);
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(T));
}

template <>
void init_int_tv<char>(shared_ptr<runtime::TensorView> tv, char min, char max)
{
    size_t size = tv->get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    vector<char> vec(size);
    for (char& element : vec)
    {
        element = static_cast<char>(dist(s_random_engine));
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(char));
}

template <>
void init_int_tv<int8_t>(shared_ptr<runtime::TensorView> tv, int8_t min, int8_t max)
{
    size_t size = tv->get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    vector<int8_t> vec(size);
    for (int8_t& element : vec)
    {
        element = static_cast<int8_t>(dist(s_random_engine));
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(int8_t));
}

template <>
void init_int_tv<uint8_t>(shared_ptr<runtime::TensorView> tv, uint8_t min, uint8_t max)
{
    size_t size = tv->get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    vector<uint8_t> vec(size);
    for (uint8_t& element : vec)
    {
        element = static_cast<uint8_t>(dist(s_random_engine));
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(uint8_t));
}

template <typename T>
void init_real_tv(shared_ptr<runtime::TensorView> tv, T min, T max)
{
    size_t size = tv->get_element_count();
    uniform_real_distribution<T> dist(min, max);
    vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(s_random_engine);
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(T));
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
                   bool timing_detail,
                   int warmup_iterations)
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
    vector<bool> args_cacheable;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto tensor = backend->create_tensor(param->get_element_type(), param->get_shape());
        random_init(tensor);
        args.push_back(tensor);
        args_cacheable.push_back(param->get_cacheable());
    }
    vector<shared_ptr<runtime::TensorView>> results;
    for (shared_ptr<Node> out : f->get_results())
    {
        auto result = backend->create_tensor(out->get_element_type(), out->get_shape());
        results.push_back(result);
    }

    for (size_t i = 0; i < args.size(); i++)
    {
        if (args_cacheable[i])
        {
            args[i]->set_stale(false);
        }
    }

    if (warmup_iterations)
    {
        for (int i = 0; i < warmup_iterations; i++)
        {
            backend->call(f, results, args);
        }
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

    if (timing_detail)
    {
        cout << "\n---- Aggregate times per op type ----\n";
        print_times(timing);

        cout << "\n---- Aggregate times per op type/shape/count ----\n";
        print_times(timing_details);
    }
}
