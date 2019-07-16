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

#include <random>

#include "benchmark.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void set_denormals_flush_to_zero();

void random_init(shared_ptr<runtime::Tensor> tensor);

std::default_random_engine& get_random_engine();

template <typename T>
void init_int_tensor(shared_ptr<runtime::Tensor> tensor, T min, T max)
{
    size_t size = tensor->get_element_count();
    uniform_int_distribution<T> dist(min, max);
    vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(get_random_engine());
    }
    tensor->write(vec.data(), vec.size() * sizeof(T));
}

template <typename T>
void init_real_tensor(shared_ptr<runtime::Tensor> tensor, T min, T max)
{
    size_t size = tensor->get_element_count();
    uniform_real_distribution<T> dist(min, max);
    vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(get_random_engine());
    }
    tensor->write(vec.data(), vec.size() * sizeof(T));
}
