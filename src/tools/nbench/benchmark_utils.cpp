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

#if defined(__x86_64__) || defined(__amd64__)
#include <xmmintrin.h>
#endif

#include "benchmark_utils.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

template <>
void init_int_tensor<char>(shared_ptr<runtime::Tensor> tensor, char min, char max)
{
    size_t size = tensor->get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    vector<char> vec(size);
    for (char& element : vec)
    {
        element = static_cast<char>(dist(get_random_engine()));
    }
    tensor->write(vec.data(), vec.size() * sizeof(char));
}

template <>
void init_int_tensor<int8_t>(shared_ptr<runtime::Tensor> tensor, int8_t min, int8_t max)
{
    size_t size = tensor->get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    vector<int8_t> vec(size);
    for (int8_t& element : vec)
    {
        element = static_cast<int8_t>(dist(get_random_engine()));
    }
    tensor->write(vec.data(), vec.size() * sizeof(int8_t));
}

template <>
void init_int_tensor<uint8_t>(shared_ptr<runtime::Tensor> tensor, uint8_t min, uint8_t max)
{
    size_t size = tensor->get_element_count();
    uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    vector<uint8_t> vec(size);
    for (uint8_t& element : vec)
    {
        element = static_cast<uint8_t>(dist(get_random_engine()));
    }
    tensor->write(vec.data(), vec.size() * sizeof(uint8_t));
}

void set_denormals_flush_to_zero()
{
#if defined(__x86_64__) || defined(__amd64__)
    // Avoids perf impact from denormals while benchmarking with random data
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}

void random_init(shared_ptr<runtime::Tensor> tensor)
{
    element::Type et = tensor->get_element_type();
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (et)
    {
    case element::Type_t::boolean: init_int_tensor<char>(tensor, 0, 1); break;
    case element::Type_t::f32: init_real_tensor<float>(tensor, -1, 1); break;
    case element::Type_t::f64: init_real_tensor<double>(tensor, -1, 1); break;
    case element::Type_t::i8: init_int_tensor<int8_t>(tensor, -1, 1); break;
    case element::Type_t::i16: init_int_tensor<int16_t>(tensor, -1, 1); break;
    case element::Type_t::i32: init_int_tensor<int32_t>(tensor, 0, 1); break;
    case element::Type_t::i64: init_int_tensor<int64_t>(tensor, 0, 1); break;
    case element::Type_t::u8: init_int_tensor<uint8_t>(tensor, 0, 1); break;
    case element::Type_t::u16: init_int_tensor<uint16_t>(tensor, 0, 1); break;
    case element::Type_t::u32: init_int_tensor<uint32_t>(tensor, 0, 1); break;
    case element::Type_t::u64: init_int_tensor<uint64_t>(tensor, 0, 1); break;
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    default: throw runtime_error("unsupported type");
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
}

default_random_engine& get_random_engine()
{
    static std::default_random_engine s_random_engine;
    return s_random_engine;
}
