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

#include <algorithm>

#include "ngraph/ngraph.hpp"
#include "ngraph/util.hpp"
#include "test_tools.hpp"

using namespace std;
using namespace ngraph;

vector<float> read_float_vector(shared_ptr<runtime::Tensor> tv)
{
    vector<float> float_vec;
    element::Type element_type = tv->get_tensor_layout()->get_element_type();

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

template <>
void init_int_tv<char>(ngraph::runtime::Tensor* tv,
                       std::default_random_engine& engine,
                       char min,
                       char max)
{
    size_t size = tv->get_element_count();
    std::uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    std::vector<char> vec(size);
    for (char& element : vec)
    {
        element = static_cast<char>(dist(engine));
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(char));
}

template <>
void init_int_tv<int8_t>(ngraph::runtime::Tensor* tv,
                         std::default_random_engine& engine,
                         int8_t min,
                         int8_t max)
{
    size_t size = tv->get_element_count();
    std::uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    std::vector<int8_t> vec(size);
    for (int8_t& element : vec)
    {
        element = static_cast<int8_t>(dist(engine));
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(int8_t));
}

template <>
void init_int_tv<uint8_t>(ngraph::runtime::Tensor* tv,
                          std::default_random_engine& engine,
                          uint8_t min,
                          uint8_t max)
{
    size_t size = tv->get_element_count();
    std::uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    std::vector<uint8_t> vec(size);
    for (uint8_t& element : vec)
    {
        element = static_cast<uint8_t>(dist(engine));
    }
    tv->write(vec.data(), 0, vec.size() * sizeof(uint8_t));
}

void random_init(ngraph::runtime::Tensor* tv, std::default_random_engine& engine)
{
    element::Type et = tv->get_element_type();
    if (et == element::boolean)
    {
        init_int_tv<char>(tv, engine, 0, 1);
    }
    else if (et == element::f32)
    {
        init_real_tv<float>(tv, engine, numeric_limits<float>::min(), 1.0f);
    }
    else if (et == element::f64)
    {
        init_real_tv<double>(tv, engine, numeric_limits<float>::min(), 1.0f);
    }
    else if (et == element::i8)
    {
        init_int_tv<int8_t>(tv, engine, -1, 1);
    }
    else if (et == element::i16)
    {
        init_int_tv<int16_t>(tv, engine, -1, 1);
    }
    else if (et == element::i32)
    {
        init_int_tv<int32_t>(tv, engine, 0, 1);
    }
    else if (et == element::i64)
    {
        init_int_tv<int64_t>(tv, engine, 0, 1);
    }
    else if (et == element::u8)
    {
        init_int_tv<uint8_t>(tv, engine, 0, 1);
    }
    else if (et == element::u16)
    {
        init_int_tv<uint16_t>(tv, engine, 0, 1);
    }
    else if (et == element::u32)
    {
        init_int_tv<uint32_t>(tv, engine, 0, 1);
    }
    else if (et == element::u64)
    {
        init_int_tv<uint64_t>(tv, engine, 0, 1);
    }
    else
    {
        throw runtime_error("unsupported type");
    }
}

template <>
void print_results(std::vector<char>& ref_data, std::vector<char>& actual_data, size_t max_results)
{
    size_t num_results = std::min(static_cast<size_t>(max_results), ref_data.size());
    std::cout << "First " << num_results << " results";
    for (size_t i = 0; i < num_results; ++i)
    {
        std::cout << "\n"
                  << std::setw(4) << i << " ref: " << std::setw(16) << std::left
                  << static_cast<int>(ref_data[i]) << "  actual: " << std::setw(16) << std::left
                  << static_cast<int>(actual_data[i]);
    }
    std::cout << std::endl;
}
