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
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/manager.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "random.hpp"

std::multimap<size_t, std::string>
    aggregate_timing(const std::vector<ngraph::runtime::PerformanceCounter>& perf_data)
{
    std::unordered_map<std::string, size_t> timing;
    for (const ngraph::runtime::PerformanceCounter& p : perf_data)
    {
        std::string op = p.name().substr(0, p.name().find('_'));
        timing[op] += p.microseconds();
    }

    std::multimap<size_t, std::string> rc;
    for (const std::pair<std::string, size_t>& t : timing)
    {
        rc.insert({t.second, t.first});
    }
    return rc;
}

void run_benchmark(const std::string& json_path, const std::string& backend_name, size_t iterations)
{
    using namespace std;
    using namespace ngraph;
    string env_var_name = "NGRAPH_" + backend_name + "_EMIT_TIMING";
    bool emit_timing = (std::getenv(env_var_name.c_str()) != nullptr);
    if (!emit_timing)
    {
        cout << "To get per-op timing set the environment variable " << env_var_name << "\n";
    }

    ngraph::test::Uniform<float> rng{-1, 1, 0};
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> f = deserialize(ss);

    stopwatch build_time;
    build_time.start();
    auto manager = runtime::Manager::get(backend_name);
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);
    build_time.stop();
    cout << "build_time " << build_time.get_milliseconds() << "ms" << endl;

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
    for (auto it = timing.rbegin(); it != timing.rend(); it++)
    {
        cout.imbue(locale(""));
        cout << setw(15) << left << it->second << " " << setw(10) << right << it->first << "us\n";
    }
}
