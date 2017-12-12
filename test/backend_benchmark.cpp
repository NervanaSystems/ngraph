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

#include <algorithm>
#include <cinttypes>
#include <cmath>

#include "gtest/gtest.h"

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/serializer.hpp"
#include "util/ndarray.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

//
// Benchmarks a graph that concatenates six 32x1x200 arrays along the middle axis.
//
TEST(benchmark, concat_32x1x200_axis1_6)
{
    const size_t n_arrays = 6;
    Shape shape_of_each_array = Shape{32, 1, 200};
    size_t concatenation_axis = 1;

    Shape result_shape;
    result_shape = shape_of_each_array;
    result_shape[concatenation_axis] *= n_arrays;

    size_t elements_per_array = 1;
    for (size_t d : shape_of_each_array)
    {
        elements_per_array *= d;
    }

    vector<vector<float>> data_arrays(n_arrays);
    for (size_t i = 0; i < n_arrays; i++)
    {
        data_arrays[i] = vector<float>(elements_per_array);
        for (size_t j = 0; j < elements_per_array; j++)
        {
            data_arrays[i][j] = float(j + 1);
        }
    }

    vector<std::string> backend_names{"INTERPRETER", "NGVM", "CPU"};
    vector<int> n_runs{200, 200, 200000};                    // one for each backend
    vector<std::function<void()>> test_callbacks;            // one for each backend
    vector<std::shared_ptr<runtime::TensorView>> result_tvs; // one for each backend

    for (std::string backend_name : backend_names)
    {
        vector<std::shared_ptr<op::Parameter>> params(n_arrays);
        vector<std::shared_ptr<Node>> params_as_nodes(n_arrays);
        for (size_t i = 0; i < n_arrays; i++)
        {
            auto param = make_shared<op::Parameter>(
                make_shared<TensorViewType>(element::Float32::element_type(), shape_of_each_array));
            params[i] = param;
            params_as_nodes[i] = param;
        }

        auto concat = make_shared<op::Concat>(params_as_nodes, concatenation_axis);
        auto f = make_shared<Function>(concat, params);

        auto manager = runtime::Manager::get(backend_name);
        auto external = manager->compile(f);
        auto backend = manager->allocate_backend();
        auto cf = backend->make_call_frame(external);

        vector<shared_ptr<runtime::Value>> input_vals;

        for (size_t i = 0; i < n_arrays; i++)
        {
            auto tv = backend->make_primary_tensor_view(element::Float32::element_type(),
                                                        shape_of_each_array);
            copy_data(tv, data_arrays[i]);
            input_vals.push_back(tv);
        }

        auto result_tv =
            backend->make_primary_tensor_view(element::Float32::element_type(), result_shape);
        result_tvs.push_back(result_tv);

        std::function<void()> cb = [input_vals, result_tv, cf]() {
            cf->call(input_vals, {result_tv});
        };

        test_callbacks.push_back(cb);
    }

    for (size_t i = 0; i < backend_names.size(); i++)
    {
        std::cout << backend_names[i] << ": " << n_runs[i] << " tests in " << std::flush;

        stopwatch sw;
        std::function<void()> cb = test_callbacks[i];

        sw.start();
        for (int j = 0; j < n_runs[i]; j++)
        {
            cb();
        }
        sw.stop();

        std::cout << sw.get_milliseconds() << "ms (" << (sw.get_microseconds() / n_runs[i])
                  << " us/test)" << std::endl;
    }

    for (size_t i = 1; i < backend_names.size(); i++)
    {
        std::cout << "Verifying " << backend_names[i] << " result against " << backend_names[0]
                  << "... " << std::flush;

        if (result_tvs[i]->get_vector<float>() == result_tvs[0]->get_vector<float>())
        {
            std::cout << " OK" << std::endl;
        }
        else
        {
            std::cout << " FAILED" << std::endl;
            ADD_FAILURE();
        }
    }
}
