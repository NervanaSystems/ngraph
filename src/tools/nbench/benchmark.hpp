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

#pragma once

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/runtime/tensor.hpp"

/// performance test utilities
std::multimap<size_t, std::string>
    aggregate_timing(const std::vector<ngraph::runtime::PerformanceCounter>& perf_data);

std::vector<ngraph::runtime::PerformanceCounter> run_benchmark(std::shared_ptr<ngraph::Function> f,
                                                               const std::string& backend_name,
                                                               size_t iterations,
                                                               bool timing_detail,
                                                               int warmup_iterations,
                                                               bool copy_data);

void run_benchmark_validation(std::shared_ptr<ngraph::Function> f, const std::string& backend_name);

template <typename T>
void copy_data(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

template <typename T>
std::vector<T> read_vector(std::shared_ptr<ngraph::runtime::Tensor> tv)
{
    if (ngraph::element::from<T>() != tv->get_tensor_layout()->get_element_type())
    {
        throw std::invalid_argument("read_vector type must match Tensor type");
    }
    size_t element_count = ngraph::shape_size(tv->get_shape());
    size_t size = element_count * sizeof(T);
    std::vector<T> rc(element_count);
    tv->read(rc.data(), 0, size);
    return rc;
}

template <typename T, typename T1 = T>
std::vector<std::vector<T1>> execute(const std::shared_ptr<ngraph::Function>& function,
                                     std::vector<std::vector<T>> args,
                                     const std::string& backend_id)
{
    auto backend = ngraph::runtime::Backend::create(backend_id);

    auto parms = function->get_parameters();

    if (parms.size() != args.size())
    {
        throw ngraph::ngraph_error("number of parameters and arguments don't match");
    }

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> arg_tensors(args.size());
    for (size_t i = 0; i < args.size(); i++)
    {
        auto t = backend->create_tensor(parms.at(i)->get_element_type(), parms.at(i)->get_shape());
        copy_data(t, args.at(i));
        arg_tensors.at(i) = t;
    }

    auto results = function->get_results();
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> result_tensors(results.size());

    for (size_t i = 0; i < results.size(); i++)
    {
        result_tensors.at(i) =
            backend->create_tensor(results.at(i)->get_element_type(), results.at(i)->get_shape());
    }

    backend->call_with_validate(function, result_tensors, arg_tensors);

    std::vector<std::vector<T1>> result_vectors;
    for (auto rt : result_tensors)
    {
        result_vectors.push_back(read_vector<T1>(rt));
    }
    return result_vectors;
}

union FloatUnion {
    float f;
    uint32_t i;
};

bool close_f(float a, float b, int mantissa_bits, int tolerance_bits);

bool all_close_f(const std::vector<float>& a,
                 const std::vector<float>& b,
                 int mantissa_bits,
                 int tolerance_bits);

/// \brief A predictable pseudo-random number generator
/// The seed is initialized so that we get repeatable pseudo-random numbers for tests
template <typename T>
class Uniform
{
public:
    Uniform(T min, T max, T seed = 0)
        : m_engine(seed)
        , m_distribution(min, max)
        , m_r(std::bind(m_distribution, m_engine))
    {
    }

    /// \brief Randomly initialize a tensor
    /// \param ptv The tensor to initialize
    const std::shared_ptr<ngraph::runtime::Tensor>
        initialize(const std::shared_ptr<ngraph::runtime::Tensor>& ptv)
    {
        std::vector<T> vec = read_vector<T>(ptv);
        initialize(vec);
        write_vector(ptv, vec);
        return ptv;
    }
    /// \brief Randomly initialize a vector
    /// \param vec The tensor to initialize
    void initialize(std::vector<T>& vec)
    {
        for (T& elt : vec)
        {
            elt = m_r();
        }
    }

protected:
    std::default_random_engine m_engine;
    std::uniform_real_distribution<T> m_distribution;
    std::function<T()> m_r;
};
