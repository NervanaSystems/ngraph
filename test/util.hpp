/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstdio>
#include <iostream>

#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
std::vector<std::vector<T>>
    execute(std::shared_ptr<Function> f, std::vector<std::vector<T>> args, std::string cbackend)
{
    auto backend = runtime::Backend::create(cbackend);

    auto parms = f->get_parameters();

    if (parms.size() != args.size())
    {
        throw ngraph_error("number of parameters and arguments don't match");
    }

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> arg_tensors(args.size());
    for (size_t i = 0; i < args.size(); i++)
    {
        auto t = backend->create_tensor(parms.at(i)->get_element_type(), parms.at(i)->get_shape());
        copy_data(t, args.at(i));
        arg_tensors.at(i) = t;
    }

    auto results = f->get_results();
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> result_tensors(results.size());

    for (size_t i = 0; i < results.size(); i++)
    {
        result_tensors.at(i) =
            backend->create_tensor(results.at(i)->get_element_type(), results.at(i)->get_shape());
    }

    backend->call(f, result_tensors, arg_tensors);

    std::vector<std::vector<T>> result_vectors;
    for (auto rt : result_tensors)
    {
        result_vectors.push_back(read_vector<T>(rt));
    }
    return result_vectors;
}
