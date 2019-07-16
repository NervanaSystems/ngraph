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
#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "benchmark_utils.hpp"

using namespace std;
using namespace ngraph;

vector<runtime::PerformanceCounter> run_benchmark_pipelined(shared_ptr<Function> f,
                                                                  const string& backend_name,
                                                                  size_t iterations,
                                                                  bool timing_detail,
                                                                  int warmup_iterations,
                                                                  bool copy_data)
{
    constexpr size_t pipeline_depth = 2;
    stopwatch timer;
    timer.start();
    auto backend = runtime::Backend::create(backend_name);
    auto exec = backend->compile(f, timing_detail);
    timer.stop();
    cout.imbue(locale(""));
    cout << "compile time: " << timer.get_milliseconds() << "ms" << endl;
    set_denormals_flush_to_zero();

    // Create random input data for all input tensors
    array<vector<shared_ptr<runtime::HostTensor>>, pipeline_depth> parameters_data_set;
    array<vector<shared_ptr<runtime::HostTensor>>, pipeline_depth> results_data_set;
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        vector<shared_ptr<runtime::HostTensor>> parameters_data;
        for (shared_ptr<op::Parameter> param : f->get_parameters())
        {
            auto tensor_data =
                make_shared<runtime::HostTensor>(param->get_element_type(), param->get_shape());
            random_init(tensor_data);
            parameters_data.push_back(tensor_data);
        }
        parameters_data_set[i] = parameters_data;
    }

    // Create input tensors for all Parameters
    array<vector<shared_ptr<runtime::Tensor>>, pipeline_depth> input_tensors_array;
    size_t input_index = 0;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto input_tensors = exec->create_input_tensor(input_index++, pipeline_depth);
        for(size_t i=0; i<pipeline_depth; i++)
        {
            input_tensors_array[i].push_back(input_tensors[i]);
        }
    }

    // Create output tensors for all Results
    array<vector<shared_ptr<runtime::Tensor>>, pipeline_depth> output_tensors_array;
    size_t output_index = 0;
    for (shared_ptr<Node> result : f->get_results())
    {
        auto output_tensors = exec->create_output_tensor(output_index++, pipeline_depth);
        for(size_t i=0; i<pipeline_depth; i++)
        {
            output_tensors_array[i].push_back(output_tensors[i]);
        }
    }

    stopwatch t1;

    // // Before we start we write the first iteration's data
    // size_t buffer_number = 0;
    // auto args = input_tensors_array[buffer_number];
    // auto args_data = parameters_data_set[buffer_number];
    // for (size_t arg_index = 0; arg_index < args.size(); arg_index++)
    // {
    //     const shared_ptr<runtime::Tensor>& arg = args[arg_index];
    //     const shared_ptr<runtime::HostTensor>& data = args_data[arg_index];
    //     arg->begin_write(data->get_data_ptr(),
    //                      data->get_element_count() * data->get_element_type().size(),
    //                      buffer_number);
    // }

    // const vector<shared_ptr<runtime::Tensor>>& results = output_tensors[buffer_number];
    // const vector<shared_ptr<runtime::HostTensor>>& results_data = results_data_set[buffer_number];
    // for (size_t i = 0; i < iterations + warmup_iterations; i++)
    // {
    //     if (i == warmup_iterations)
    //     {
    //         t1.start();
    //     }
    //     future<void> exec_future = exec->begin_execute(results, args);
    //     if (i > 0)
    //     {
    //         for (size_t result_index = 0; result_index < results.size(); result_index++)
    //         {
    //             const shared_ptr<runtime::HostTensor>& data = results_data[result_index];
    //             const shared_ptr<runtime::Tensor>& result = results[result_index];
    //             result->begin_read(data->get_data_ptr(),
    //                                data->get_element_count() * data->get_element_type().size(),
    //                                (buffer_number - 1) & 1);
    //         }
    //     }
    //     buffer_number = (buffer_number + 1) & 1;
    //     for (size_t arg_index = 0; arg_index < args.size(); arg_index++)
    //     {
    //         const shared_ptr<runtime::Tensor>& arg = args[arg_index];
    //         const shared_ptr<runtime::HostTensor>& data = args_data[arg_index];
    //         arg->begin_write(data->get_data_ptr(),
    //                          data->get_element_count() * data->get_element_type().size(),
    //                          buffer_number);
    //     }
    //     exec_future.get();
    // }
    // for (size_t result_index = 0; result_index < results.size(); result_index++)
    // {
    //     const shared_ptr<runtime::HostTensor>& data = results_data[result_index];
    //     const shared_ptr<runtime::Tensor>& result = results[result_index];
    //     result->begin_read(data->get_data_ptr(),
    //                        data->get_element_count() * data->get_element_type().size(),
    //                        (buffer_number - 1) & 1);
    // }
    // t1.stop();
    // float time = t1.get_milliseconds();
    // cout << time / iterations << "ms per iteration" << endl;

    vector<runtime::PerformanceCounter> perf_data = exec->get_performance_data();
    return perf_data;
}
