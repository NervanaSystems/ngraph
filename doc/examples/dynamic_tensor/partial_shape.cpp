//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include <numeric>
#include <vector>

#include <ngraph/ngraph.hpp>

using namespace std;
using namespace ngraph;

void execute(shared_ptr<runtime::Backend> be,
             shared_ptr<runtime::Executable> ex,
             shared_ptr<runtime::Tensor> t_out,
             uint32_t n);

int main()
{
    // Create and compile a graph where the provided info of shape of x is
    // (2,?)
    auto x_shape_info = PartialShape{2, Dimension::dynamic()};
    auto x = make_shared<op::Parameter>(element::i32, x_shape_info);
    auto a = x + x;
    auto f = make_shared<Function>(OutputVector{a}, ParameterVector{x});
    auto be = runtime::Backend::create("CPU", true);
    auto ex = be->compile(f);

    // Create a dynamic tensor of shape (2,?)
    auto t_out = be->create_dynamic_tensor(element::i32, x_shape_info);
    execute(be, ex, t_out, 3);
    execute(be, ex, t_out, 11);
    execute(be, ex, t_out, 20);

    return 0;
}

void execute(shared_ptr<runtime::Backend> be,
             shared_ptr<runtime::Executable> ex,
             shared_ptr<runtime::Tensor> t_out,
             uint32_t n)
{
    // Initialize input of shape (2, n)
    auto t_in = be->create_tensor(element::i32, Shape{2, n});
    {
        vector<int32_t> t_val(2 * n);
        iota(t_val.begin(), t_val.end(), 0);
        t_in->write(&t_val[0], t_val.size() * sizeof(t_val[0]));
    }
    // Get the result
    ex->call({t_out}, {t_in});

    auto s = t_out->get_shape();
    vector<int32_t> r(s[0] * s[1]);
    t_out->read(&r[0], r.size() * sizeof(r[0]));
    cout << "[" << endl;
    for (size_t i = 0; i < s[0]; ++i)
    {
        cout << " [";
        for (size_t j = 0; j < s[1]; ++j)
        {
            cout << r[i * s[1] + j] << ' ';
        }
        cout << ']' << endl;
    }
    cout << ']' << endl;
}
