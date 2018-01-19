// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <exception>
#include <list>
#include <memory>

#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    class Node;
    class Function;
}

bool validate_list(const std::list<std::shared_ptr<ngraph::Node>>& nodes);
std::shared_ptr<ngraph::Function> make_test_graph();

template <typename T>
void copy_data(std::shared_ptr<ngraph::runtime::TensorView> tv, const std::vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

template <typename T>
std::vector<T> get_vector(std::shared_ptr<ngraph::runtime::TensorView> tv)
{
    if (ngraph::element::from<T>() != tv->get_tensor_view_layout()->get_element_type())
    {
        throw std::invalid_argument("get_vector type must match TensorView type");
    }
    size_t element_count = ngraph::shape_size(tv->get_shape());
    size_t size = element_count * sizeof(T);
    std::vector<T> rc(element_count);
    tv->read(rc.data(), 0, size);
    return rc;
}

template <typename T>
void set_vector(std::shared_ptr<ngraph::runtime::TensorView> tv, const std::vector<T>& values)
{
    tv->write(values.data(), 0, values.size() * sizeof(T));
}
