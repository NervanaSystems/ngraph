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

#include <exception>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <random>

#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"

namespace ngraph
{
    class Node;
    class Function;
}

bool validate_list(const std::list<std::shared_ptr<ngraph::Node>>& nodes);
std::shared_ptr<ngraph::Function> make_test_graph();

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

std::vector<float> read_float_vector(std::shared_ptr<ngraph::runtime::Tensor> tv);

template <typename T>
void write_vector(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<T>& values)
{
    tv->write(values.data(), 0, values.size() * sizeof(T));
}

template <typename T>
std::vector<std::shared_ptr<T>> get_ops_of_type(std::shared_ptr<ngraph::Function> f)
{
    std::vector<std::shared_ptr<T>> ops;
    for (auto op : f->get_ops())
    {
        if (auto cop = std::dynamic_pointer_cast<T>(op))
        {
            ops.push_back(cop);
        }
    }

    return ops;
}

template <typename T>
size_t count_ops_of_type(std::shared_ptr<ngraph::Function> f)
{
    size_t count = 0;
    for (auto op : f->get_ops())
    {
        if (std::dynamic_pointer_cast<T>(op))
        {
            count++;
        }
    }

    return count;
}

template <typename T>
void init_int_tv(ngraph::runtime::Tensor* tv, std::default_random_engine& engine, T min, T max)
{
    size_t size = tv->get_element_count();
    std::uniform_int_distribution<T> dist(min, max);
    std::vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(engine);
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(T));
}

template <typename T>
void init_real_tv(ngraph::runtime::Tensor* tv, std::default_random_engine& engine, T min, T max)
{
    size_t size = tv->get_element_count();
    std::uniform_real_distribution<T> dist(min, max);
    std::vector<T> vec(size);
    for (T& element : vec)
    {
        element = dist(engine);
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(T));
}

void random_init(ngraph::runtime::Tensor* tv, std::default_random_engine& engine);

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

template <typename T>
void print_results(std::vector<T>& ref_data, std::vector<T>& actual_data, size_t max_results = 16)
{
    size_t num_results = std::min(static_cast<size_t>(max_results), ref_data.size());
    std::cout << "First " << num_results << " results";
    for (size_t i = 0; i < num_results; ++i)
    {
        std::cout << "\n"
                  << std::setw(4) << i << " ref: " << std::setw(16) << std::left << ref_data[i]
                  << "  actual: " << std::setw(16) << std::left << actual_data[i];
    }
    std::cout << std::endl;
}

template <>
void print_results(std::vector<char>& ref_data, std::vector<char>& actual_data, size_t max_results);
