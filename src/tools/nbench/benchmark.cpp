//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <random>
#include <xmmintrin.h>

#include "benchmark.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static default_random_engine s_random_engine;

void set_denormals_flush_to_zero()
{
#if defined(__x86_64__) || defined(__amd64__)
    // Avoids perf impact from denormals while benchmarking with random data
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

template <typename T>
void init_int_tv(shared_ptr<runtime::Tensor> tv, T min, T max)
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
void init_int_tv<char>(shared_ptr<runtime::Tensor> tv, char min, char max)
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
void init_int_tv<int8_t>(shared_ptr<runtime::Tensor> tv, int8_t min, int8_t max)
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
void init_int_tv<uint8_t>(shared_ptr<runtime::Tensor> tv, uint8_t min, uint8_t max)
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
void init_real_tv(shared_ptr<runtime::Tensor> tv, T min, T max)
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

static void random_init(shared_ptr<runtime::Tensor> tv)
{
    element::Type et = tv->get_element_type();
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
        init_int_tv<int32_t>(tv, 0, 1);
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
    backend->enable_performance_data(f, timing_detail);
    backend->compile(f);
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
        random_init(tensor);
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

    if (warmup_iterations)
    {
        for (int i = 0; i < warmup_iterations; i++)
        {
            backend->call(f, results, args);
        }
    }

    stopwatch t1;
    t1.start();
    for (size_t i = 0; i < iterations; i++)
    {
        if (copy_data)
        {
            for (size_t arg_index = 0; arg_index < args.size(); arg_index++)
            {
                const shared_ptr<runtime::Tensor>& arg = args[arg_index];
                if (arg->get_stale())
                {
                    const shared_ptr<runtime::HostTensor>& data = arg_data[arg_index];
                    arg->write(data->get_data_ptr(),
                               0,
                               data->get_element_count() * data->get_element_type().size());
                }
            }
        }
        backend->call(f, results, args);
        if (copy_data)
        {
            for (size_t result_index = 0; result_index < results.size(); result_index++)
            {
                const shared_ptr<runtime::HostTensor>& data = result_data[result_index];
                const shared_ptr<runtime::Tensor>& result = results[result_index];
                result->read(data->get_data_ptr(),
                             0,
                             data->get_element_count() * data->get_element_type().size());
            }
        }
    }
    t1.stop();
    float time = t1.get_milliseconds();
    cout << time / iterations << "ms per iteration" << endl;

    vector<runtime::PerformanceCounter> perf_data = backend->get_performance_data(f);
    return perf_data;
}

void run_benchmark_validation(shared_ptr<Function> f, const string& backend_name)
{
    NodeVector new_results;
    for (auto n : f->get_ordered_ops())
    {
        //dont include op::Results otherwise Function c-tor will complain
        if (!n->is_output() && !n->is_parameter() && !n->is_constant() &&
            !(n->get_outputs().size() > 1) && n->get_element_type() == element::f32)
        {
            // place conditionals here if you want to only make certain ops an output/result node
            new_results.push_back(n);
        }
    }

    //no need to include original results they are subsumed by new_results
    auto new_func = make_shared<Function>(new_results, f->get_parameters());

    // // uncomment these lines to serialize the new_func for later use
    // // I use this for splicing a small graph out of a larger one
    // string js = serialize(new_func, 4);
    // std::ofstream outfile;
    // outfile.open("conv_bprop_filters.json");
    // outfile << js;
    // outfile.close();
    // if (new_func) exit(0);

    Uniform<float> rng(1.0f, 2.0f, 2112);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : new_func->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto cpu_func = ngraph::clone_function(*new_func);
    auto bk_func = ngraph::clone_function(*new_func);

    auto cpu_results = execute(cpu_func, args, "CPU");
    auto bk_results = execute(bk_func, args, backend_name);
    for (size_t i = 0; i < cpu_results.size(); i++)
    {
        std::cout << "Comparing results for " << new_results.at(i)->get_name() << std::endl;
        if (auto node = dynamic_pointer_cast<op::GetOutputElement>(new_results.at(i)))
        {
            std::cout << "  Parent node: ";
            for (auto& p : node->get_arguments())
            {
                std::cout << " " << p->get_name() << std::endl;
                std::cout << "   nargs: " << p->get_arguments().size() << std::endl;
            }
        }
        all_close_f(cpu_results.at(i), bk_results.at(i), 24, 0);
    }

    cout << "validation for" << f->get_name() << " done." << endl;
    return;
}

bool close_f(float a, float b, int mantissa_bits, int tolerance_bits)
{
    // isfinite(a) => !isinf(a) && !isnan(a)
    if (!std::isfinite(a) || !std::isfinite(b))
    {
        return false;
    }

    FloatUnion a_fu{a};
    FloatUnion b_fu{b};
    uint32_t a_uint = a_fu.i;
    uint32_t b_uint = b_fu.i;

    // A trick to handle both positive and negative numbers, see https://goo.gl/YbdnFQ
    // - If negative: convert to two's complement
    // - If positive: mask with sign bit
    uint32_t sign_mask = static_cast<uint32_t>(1U) << 31;
    a_uint = (sign_mask & a_uint) ? (~a_uint + 1) : (sign_mask | a_uint);
    b_uint = (sign_mask & b_uint) ? (~b_uint + 1) : (sign_mask | b_uint);

    uint32_t distance = (a_uint >= b_uint) ? (a_uint - b_uint) : (b_uint - a_uint);

    // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
    // tolerance_bit_shift = 32 -           (1 +  8 + (24 -     1         ) - 2             )
    //                       float_length    sign exp  mantissa implicit 1    tolerance_bits
    uint32_t tolerance_bit_shift = 32 - (1 + 8 + (mantissa_bits - 1) - tolerance_bits);
    uint32_t tolerance = static_cast<uint32_t>(1U) << tolerance_bit_shift;

    return distance <= tolerance;
}

bool all_close_f(const std::vector<float>& a,
                 const std::vector<float>& b,
                 int mantissa_bits,
                 int tolerance_bits)
{
    bool rc = true;
    if (a.size() != b.size())
    {
        throw ngraph::ngraph_error("a.size() != b.size() for all_close comparison.");
    }
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        bool is_close_f = close_f(a[i], b[i], mantissa_bits, tolerance_bits);
        if (!is_close_f)
        {
            if (count < 5)
            {
                NGRAPH_INFO << a[i] << " is not close to " << b[i] << " at idx " << i;
            }
            rc = false;
            count++;
        }
    }

    if (!rc)
    {
        NGRAPH_INFO << "diff count " << count << " out of " << a.size();
    }
    return rc;
}
