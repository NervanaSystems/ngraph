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

#include <iostream>

#include <ngraph/ngraph.hpp>

using namespace ngraph;

int main()
{
    // Create and compile a graph where the provided info of shape of x is
    // (2,?)
    auto x_shape_info = PartialShape{2, Dimension::dynamic()};
    auto x = make_shared<op::Parameter>(element::i32, x_shape_info);
    auto a = x + x;
    auto f = make_shared<Function>({a}, {x});
    auto be = runtime::backend::create();
    auto ex = be->compile(f);

    // Create a dynamic tensor of shape (2,?)
    auto t_out = be->create_dynamic_tensor(element::i32, x_shape_info);

    // Call the graph to write a value with shape (2,3) to t_out
    auto t_in = be->create_tensor(element::i32, Shape{2, 3});
    t_in->write();
    ex->call({t_out}, {t_in})

        // Call the graph again, to write a value with a different shape to
        // t_out.
        t_in = be->create_tensor(element::i32, Shape{2, 20});
    t_in->write();
    ex->call({t_out}, {t_in})

        // Get the result. At this point t_out->get_shape() would return
        // Shape{2,20},
        // but t_out->get_partial_shape() would return "(2,?)"

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
