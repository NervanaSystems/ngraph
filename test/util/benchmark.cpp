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
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "random.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> find_node(const string& name, shared_ptr<Function> func)
{
    static unordered_map<string, shared_ptr<Node>> node_map;
    if (node_map.empty())
    {
        vector<shared_ptr<Function>> fs;
        traverse_functions(func, [&](shared_ptr<Function> f) { fs.push_back(f); });
        for (shared_ptr<Function> f : fs)
        {
            for (shared_ptr<Node> node : f->get_ops())
            {
                node_map.insert({node->get_name(), node});
            }
        }
    }
    return node_map[name];
}

multimap<size_t, string>
    aggregate_timing_details(const vector<runtime::PerformanceCounter>& perf_data,
                             shared_ptr<Function> f)
{
    unordered_map<string, size_t> timing;
    for (const runtime::PerformanceCounter& p : perf_data)
    {
        shared_ptr<Node> node = find_node(p.name(), f);
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

void run_benchmark(shared_ptr<Function> f,
                   const string& backend_name,
                   size_t iterations,
                   bool timing_detail)
{
    test::Uniform<float> rng{-1, 1, 0};

    stopwatch timer;
    timer.start();
    auto manager = runtime::Manager::get(backend_name);
    auto external = manager->compile(f);
    external->set_emit_timing(timing_detail);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    timer.stop();
    cout.imbue(locale(""));
    cout << "compile time: " << timer.get_milliseconds() << "ms" << endl;

    vector<shared_ptr<runtime::TensorView>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto tensor =
            backend->make_primary_tensor_view(param->get_element_type(), param->get_shape());
        rng.initialize(tensor);
        args.push_back(tensor);
    }
    vector<shared_ptr<runtime::TensorView>> results;
    for (shared_ptr<Node> out : f->get_results())
    {
        auto result = backend->make_primary_tensor_view(out->get_element_type(), out->get_shape());
        results.push_back(result);
    }

    stopwatch t1;
    t1.start();
    for (size_t i = 0; i < static_cast<size_t>(iterations); i++)
    {
        cf->tensor_call(args, results);
    }
    t1.stop();
    float time = t1.get_milliseconds();
    cout << time / iterations << "ms per iteration" << endl;

    vector<runtime::PerformanceCounter> perf_data = cf->get_performance_data();
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
