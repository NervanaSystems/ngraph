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

#include <algorithm>

#include "ngraph/ngraph.hpp"
#include "ngraph/util.hpp"
#include "test_tools.hpp"

using namespace std;
using namespace ngraph;

vector<float> read_float_vector(shared_ptr<runtime::TensorView> tv)
{
    vector<float> float_vec;
    element::Type element_type = tv->get_tensor_view_layout()->get_element_type();

    if (element_type == element::boolean)
    {
        vector<char> vec = read_vector<char>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::f32)
    {
        vector<float> vec = read_vector<float>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::f64)
    {
        vector<double> vec = read_vector<double>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::i8)
    {
        vector<int8_t> vec = read_vector<int8_t>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::i16)
    {
        vector<int16_t> vec = read_vector<int16_t>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::i32)
    {
        vector<int32_t> vec = read_vector<int32_t>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::i64)
    {
        vector<int64_t> vec = read_vector<int64_t>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::u8)
    {
        vector<uint8_t> vec = read_vector<uint8_t>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::u16)
    {
        vector<uint16_t> vec = read_vector<uint16_t>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::u32)
    {
        vector<uint32_t> vec = read_vector<uint32_t>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else if (element_type == element::u64)
    {
        vector<uint64_t> vec = read_vector<uint64_t>(tv);
        float_vec = vector<float>(vec.begin(), vec.end());
    }
    else
    {
        throw ngraph_error("Unsupported nGraph element type.");
    }

    return float_vec;
}

// This function traverses the list of ops and verifies that each op's dependencies (its inputs)
// is located earlier in the list. That is enough to be valid
bool validate_list(const list<shared_ptr<Node>>& nodes)
{
    bool rc = true;
    for (auto it = nodes.rbegin(); it != nodes.rend(); it++)
    {
        auto node_tmp = *it;
        auto dependencies_tmp = node_tmp->get_arguments();
        vector<Node*> dependencies;

        for (shared_ptr<Node> n : dependencies_tmp)
        {
            dependencies.push_back(n.get());
        }
        auto tmp = it;
        for (tmp++; tmp != nodes.rend(); tmp++)
        {
            auto dep_tmp = *tmp;
            auto found = find(dependencies.begin(), dependencies.end(), dep_tmp.get());
            if (found != dependencies.end())
            {
                dependencies.erase(found);
            }
        }
        if (dependencies.size() > 0)
        {
            rc = false;
        }
    }
    return rc;
}

shared_ptr<Function> make_test_graph()
{
    auto arg_0 = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg_1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg_2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg_3 = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg_4 = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg_5 = make_shared<op::Parameter>(element::f32, Shape{});

    auto t0 = make_shared<op::Add>(arg_0, arg_1);
    auto t1 = make_shared<op::Dot>(t0, arg_2);
    auto t2 = make_shared<op::Multiply>(t0, arg_3);

    auto t3 = make_shared<op::Add>(t1, arg_4);
    auto t4 = make_shared<op::Add>(t2, arg_5);

    auto r0 = make_shared<op::Add>(t3, t4);

    auto f0 =
        make_shared<Function>(r0, op::ParameterVector{arg_0, arg_1, arg_2, arg_3, arg_4, arg_5});

    return f0;
}
