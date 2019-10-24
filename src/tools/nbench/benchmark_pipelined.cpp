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

#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>

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

class TensorCollection
{
public:
    vector<shared_ptr<runtime::HostTensor>> parameter_data;
    vector<shared_ptr<runtime::HostTensor>> result_data;

    vector<shared_ptr<runtime::Tensor>> input_tensors;
    vector<shared_ptr<runtime::Tensor>> output_tensors;

private:
};

static mutex s_mutex;
static condition_variable s_condition;
static size_t current_iteration = 0;
static size_t s_iterations;
static size_t s_warmup_iterations;
static stopwatch s_timer;

static void
    thread_entry(runtime::Executable* exec, const TensorCollection& tensors, size_t pipeline_stage)
{
    bool data_written = false;
    const vector<shared_ptr<runtime::Tensor>>& args = tensors.input_tensors;
    const vector<shared_ptr<runtime::Tensor>>& results = tensors.output_tensors;
    while (current_iteration < s_iterations + s_warmup_iterations)
    {
        if (!data_written)
        {
            for (size_t arg_index = 0; arg_index < args.size(); arg_index++)
            {
                const shared_ptr<runtime::Tensor>& arg = args[arg_index];
                if (arg->get_stale())
                {
                    const shared_ptr<runtime::HostTensor>& data = tensors.parameter_data[arg_index];
                    arg->write(data->get_data_ptr(),
                               data->get_element_count() * data->get_element_type().size());
                }
            }
            data_written = true;
        }
        unique_lock<mutex> lock(s_mutex);
        if ((current_iteration & 1) != pipeline_stage)
        {
            s_condition.wait(lock);
        }
        else
        {
            if (current_iteration == s_warmup_iterations)
            {
                s_timer.start();
            }
            // our turn to run
            exec->call(results, args);
            current_iteration++;
            data_written = false;
            s_condition.notify_all();
            lock.unlock();
            for (size_t result_index = 0; result_index < results.size(); result_index++)
            {
                const shared_ptr<runtime::HostTensor>& data = tensors.result_data[result_index];
                const shared_ptr<runtime::Tensor>& result = results[result_index];
                result->read(data->get_data_ptr(),
                             data->get_element_count() * data->get_element_type().size());
            }
            if (current_iteration == (s_iterations + s_warmup_iterations - 1))
            {
                s_timer.stop();
            }
        }
    }
}

vector<runtime::PerformanceCounter> run_benchmark_pipelined(shared_ptr<Function> f,
                                                            const string& backend_name,
                                                            size_t iterations,
                                                            bool timing_detail,
                                                            int warmup_iterations,
                                                            bool /* copy_data */)
{
    constexpr size_t pipeline_depth = 2;
    s_iterations = iterations;
    s_warmup_iterations = warmup_iterations;
    array<TensorCollection, pipeline_depth> tensor_collections;
    stopwatch timer;
    timer.start();
    auto backend = runtime::Backend::create(backend_name);
    auto exec = backend->compile(f, timing_detail);
    timer.stop();
    cout.imbue(locale(""));
    cout << "compile time: " << timer.get_milliseconds() << "ms" << endl;
    set_denormals_flush_to_zero();

    // Create random input data for all input tensors
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        for (shared_ptr<op::Parameter> param : f->get_parameters())
        {
            auto tensor_data =
                make_shared<runtime::HostTensor>(param->get_element_type(), param->get_shape());
            random_init(tensor_data);
            tensor_collections[i].parameter_data.push_back(tensor_data);
        }
    }

    // Create output tensors for all outputs
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        for (shared_ptr<Node> result : f->get_results())
        {
            auto tensor_data =
                make_shared<runtime::HostTensor>(result->get_element_type(), result->get_shape());
            tensor_collections[i].result_data.push_back(tensor_data);
        }
    }

    // Create input tensors for all Parameters
    array<vector<shared_ptr<runtime::Tensor>>, pipeline_depth> input_tensors_array;
    size_t input_index = 0;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto input_tensors = exec->create_input_tensor(input_index++, pipeline_depth);
        for (size_t i = 0; i < pipeline_depth; i++)
        {
            tensor_collections[i].input_tensors.push_back(input_tensors[i]);
        }
    }

    // Create output tensors for all Results
    array<vector<shared_ptr<runtime::Tensor>>, pipeline_depth> output_tensors_array;
    size_t output_index = 0;
    for (shared_ptr<Node> result : f->get_results())
    {
        auto output_tensors = exec->create_output_tensor(output_index++, pipeline_depth);
        for (size_t i = 0; i < pipeline_depth; i++)
        {
            tensor_collections[i].output_tensors.push_back(output_tensors[i]);
        }
    }

    thread threads[pipeline_depth];
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        threads[i] = thread(thread_entry, exec.get(), tensor_collections[i], i);
    }

    for (size_t i = 0; i < pipeline_depth; i++)
    {
        threads[i].join();
    }
    float time = s_timer.get_milliseconds();
    cout << time / iterations << "ms per iteration" << endl;

    vector<runtime::PerformanceCounter> perf_data = exec->get_performance_data();
    return perf_data;
}
