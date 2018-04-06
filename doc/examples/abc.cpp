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

#include <iostream>

#include <ngraph/ngraph.hpp>

using namespace ngraph;

int main()
{
    // Build the graph
    Shape s{2, 3};
    auto a = std::make_shared<op::Parameter>(element::f32, s);
    auto b = std::make_shared<op::Parameter>(element::f32, s);
    auto c = std::make_shared<op::Parameter>(element::f32, s);

    auto t0 = std::make_shared<op::Add>(a, b);
    auto t1 = std::make_shared<op::Multiply>(t0, c);

    // Make the function
    auto f = std::make_shared<Function>(NodeVector{t1},
                                        op::ParameterVector{a, b, c});

    // Get the backend
    auto manager = runtime::Manager::get("CPU");
    auto backend = manager->allocate_backend();

    // Compile the function
    auto external = manager->compile(f);
    auto cf = backend->make_call_frame(external);

    // Allocate tensors for arguments a, b, c
    auto t_a = backend->make_primary_tensor_view(element::f32, s);
    auto t_b = backend->make_primary_tensor_view(element::f32, s);
    auto t_c = backend->make_primary_tensor_view(element::f32, s);
    // Allocate tensor for the result
    auto t_result = backend->make_primary_tensor_view(element::f32, s);

    // Initialize tensors
    float v_a[2][3] = {{1, 2, 3}, {4, 5, 6}};
    float v_b[2][3] = {{7, 8, 9}, {10, 11, 12}};
    float v_c[2][3] = {{1, 0, -1}, {-1, 1, 2}};

    t_a->write(&v_a, 0, sizeof(v_a));
    t_b->write(&v_b, 0, sizeof(v_b));
    t_c->write(&v_c, 0, sizeof(v_c));

    // Invoke the function
    cf->call({t_result}, {t_a, t_b, t_c});

    // Get the result
    float r[2][3];
    t_result->read(&r, 0, sizeof(r));

    std::cout << "[" << std::endl;
    for (size_t i = 0; i < s[0]; ++i)
    {
        std::cout << " [";
        for (size_t j = 0; j < s[1]; ++j)
        {
            std::cout << r[i][j] << ' ';
        }
        std::cout << ']' << std::endl;
    }
    std::cout << ']' << std::endl;

    return 0;
}
