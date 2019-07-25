//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "benchmark.hpp"
#include "benchmark_utils.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

vector<runtime::PerformanceCounter> run_benchmark(shared_ptr<Function> f,
                                                  const string& backend_name,
                                                  size_t iterations,
                                                  bool timing_detail,
                                                  int warmup_iterations,
                                                  bool copy_data)
{
    stopwatch timer;
    timer.start();
    auto backend = runtime::Backend::create(backend_name);
    auto exec = backend->compile(f, timing_detail);
    timer.stop();
    cout.imbue(locale(""));
    cout << "compile time: " << timer.get_milliseconds() << "ms" << endl;

    vector<shared_ptr<runtime::HostTensor>> arg_data;
    vector<shared_ptr<runtime::Tensor>> args;
    vector<bool> args_cacheable;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto tensor = backend->create_tensor(param->get_element_type(), param->get_shape());
        auto tensor_data =
            make_shared<runtime::HostTensor>(param->get_element_type(), param->get_shape());
        random_init(tensor_data);
        tensor->write(tensor_data->get_data_ptr(),
                      tensor_data->get_element_count() * tensor_data->get_element_type().size());
        args.push_back(tensor);
        arg_data.push_back(tensor_data);
        args_cacheable.push_back(param->get_cacheable());
    }
    set_denormals_flush_to_zero();

    vector<shared_ptr<runtime::HostTensor>> result_data;
    vector<shared_ptr<runtime::Tensor>> results;
    for (shared_ptr<Node> out : f->get_results())
    {
        auto result = backend->create_tensor(out->get_element_type(), out->get_shape());
        auto tensor_data =
            make_shared<runtime::HostTensor>(out->get_element_type(), out->get_shape());
        results.push_back(result);
        result_data.push_back(tensor_data);
    }

    for (size_t i = 0; i < args.size(); i++)
    {
        if (args_cacheable[i])
        {
            args[i]->set_stale(false);
        }
    }

    stopwatch t1;
    for (size_t i = 0; i < iterations + warmup_iterations; i++)
    {
        if (i == warmup_iterations)
        {
            t1.start();
        }
        if (copy_data)
        {
            for (size_t arg_index = 0; arg_index < args.size(); arg_index++)
            {
                const shared_ptr<runtime::Tensor>& arg = args[arg_index];
                if (arg->get_stale())
                {
                    const shared_ptr<runtime::HostTensor>& data = arg_data[arg_index];
                    arg->write(data->get_data_ptr(),
                               data->get_element_count() * data->get_element_type().size());
                }
            }
        }
        exec->call(results, args);
        if (copy_data)
        {
            for (size_t result_index = 0; result_index < results.size(); result_index++)
            {
                const shared_ptr<runtime::HostTensor>& data = result_data[result_index];
                const shared_ptr<runtime::Tensor>& result = results[result_index];
                result->read(data->get_data_ptr(),
                             data->get_element_count() * data->get_element_type().size());
            }
        }
    }
    t1.stop();
    float time = t1.get_milliseconds();
    cout << time / iterations << "ms per iteration" << endl;

    vector<runtime::PerformanceCounter> perf_data = exec->get_performance_data();
    return perf_data;
}
