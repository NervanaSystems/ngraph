//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

/// Utility to dump single HostTensor to std::out.
template <class TYPE>
static void dump_tensor_elements(runtime::HostTensor& tensor)
{
    TYPE* data_ptr = tensor.get_data_ptr<TYPE>();
    size_t numElements = tensor.get_element_count();
    for (size_t i = 0; i < numElements; ++i)
    {
        cout << data_ptr[i] << " ";
    }
    cout << endl;
}

/// Utility to dump all the result tensors to std::out.
static void dump_result_tensors(vector<shared_ptr<runtime::HostTensor>>& results)
{
    cout << "============================================================================\n";
    cout << "---- Dumping result tensors \n";
    cout << "============================================================================\n";

    unsigned i = 0;
    for (auto& result : results)
    {
        cout << "Result tensor #" << i << ": " << result->get_name() << endl;
        auto element_type = result->get_element_type();

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
        switch (element_type)
        {
        case (element::Type_t::f32): dump_tensor_elements<float>(*result); break;
        case (element::Type_t::u8): dump_tensor_elements<uint8_t>(*result); break;
        case (element::Type_t::i8): dump_tensor_elements<int8_t>(*result); break;
        case (element::Type_t::u16): dump_tensor_elements<uint16_t>(*result); break;
        case (element::Type_t::i16): dump_tensor_elements<int16_t>(*result); break;
        case (element::Type_t::i32): dump_tensor_elements<int32_t>(*result); break;
        case (element::Type_t::i64): dump_tensor_elements<int64_t>(*result); break;
        case (element::Type_t::f64): dump_tensor_elements<double>(*result); break;
        case (element::Type_t::u32): dump_tensor_elements<uint32_t>(*result); break;
        case (element::Type_t::u64): dump_tensor_elements<uint64_t>(*result); break;
        case (element::Type_t::boolean): dump_tensor_elements<char>(*result); break;
        case (element::Type_t::bf16): dump_tensor_elements<bfloat16>(*result); break;
        case (element::Type_t::f16): dump_tensor_elements<float16>(*result); break;
        case element::Type_t::u1: throw runtime_error("unsupported type");
        case element::Type_t::undefined: throw runtime_error("unsupported type");
        case element::Type_t::dynamic: throw runtime_error("unsupported type");
        default: NGRAPH_UNREACHABLE("Type not implemented yet");
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
        ++i;
    }
}

vector<runtime::PerformanceCounter> run_benchmark(shared_ptr<Function> f,
                                                  const string& backend_name,
                                                  size_t iterations,
                                                  bool timing_detail,
                                                  size_t warmup_iterations,
                                                  bool copy_data,
                                                  bool dump_results)
{
    stopwatch timer;
    timer.start();
    auto backend = runtime::Backend::create(backend_name);
    auto exec = backend->compile(f, timing_detail);
    timer.stop();
    stringstream ss;
    ss.imbue(locale(""));
    ss << "compile time: " << timer.get_milliseconds() << "ms" << endl;

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
    ss << time / iterations << "ms per iteration" << endl;
    cout << ss.str();

    if (dump_results)
    {
        dump_result_tensors(result_data);
    }

    vector<runtime::PerformanceCounter> perf_data = exec->get_performance_data();
    return perf_data;
}
