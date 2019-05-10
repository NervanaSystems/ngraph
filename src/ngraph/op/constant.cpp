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

#include <cmath>
#include <cstdio>

#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
string to_cpp_string(T value)
{
    string rc;
    if (std::isnan(value))
    {
        rc = "NAN";
    }
    else if (std::isinf(value))
    {
        rc = (value > 0 ? "INFINITY" : "-INFINITY");
    }
    else
    {
        stringstream ss;
        ss << value;
        rc = ss.str();
    }
    return rc;
}

op::Constant::~Constant()
{
}

vector<string> op::Constant::get_value_strings() const
{
    vector<string> rc;

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
    switch (get_element_type().get_type_enum())
    {
    case element::Type_t::boolean:
        for (int value : get_vector<char>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::bf16:
        for (bfloat16 value : get_vector<bfloat16>())
        {
            rc.push_back(to_cpp_string(static_cast<float>(value)));
        }
        break;
    case element::Type_t::f16:
        for (float16 value : get_vector<float16>())
        {
            rc.push_back(to_cpp_string(static_cast<float>(value)));
        }
        break;
    case element::Type_t::f32:
        for (float value : get_vector<float>())
        {
            rc.push_back(to_cpp_string(value));
        }
        break;
    case element::Type_t::f64:
        for (double value : get_vector<double>())
        {
            rc.push_back(to_cpp_string(value));
        }
        break;
    case element::Type_t::i8:
        for (int value : get_vector<int8_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::i16:
        for (int value : get_vector<int16_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::i32:
        for (int32_t value : get_vector<int32_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::i64:
        for (int64_t value : get_vector<int64_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::u8:
        for (uint32_t value : get_vector<uint8_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::u16:
        for (uint32_t value : get_vector<uint16_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::u32:
        for (uint32_t value : get_vector<uint32_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::u64:
        for (uint64_t value : get_vector<uint64_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::undefined: throw runtime_error("unsupported type");
    case element::Type_t::dynamic: throw runtime_error("unsupported type");
    }
#pragma GCC diagnostic pop

    return rc;
}

shared_ptr<Node> op::Constant::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Constant>(m_element_type, m_shape, m_data->get_ptr());
}

template <typename T>
static bool test_bitwise_identical(const op::Constant* constant)
{
    const size_t size = shape_size(constant->get_shape());
    bool data_is_constant = true;
    if (size > 0)
    {
        const T* data = constant->get_data_ptr<T>();
        const T compare = data[0];
        for (size_t i = 1; i < size; i++)
        {
            if (data[i] != compare)
            {
                data_is_constant = false;
                break;
            }
        }
    }
    return data_is_constant;
}

bool op::Constant::are_all_data_elements_bitwise_identical() const
{
    bool rc = false;
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
    switch (get_element_type().get_type_enum())
    {
    case element::Type_t::boolean:
    case element::Type_t::i8:
    case element::Type_t::u8:
    {
        rc = test_bitwise_identical<uint8_t>(this);
        break;
    }
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::i16:
    case element::Type_t::u16:
    {
        rc = test_bitwise_identical<uint16_t>(this);
        break;
    }
    case element::Type_t::f32:
    case element::Type_t::i32:
    case element::Type_t::u32:
    {
        rc = test_bitwise_identical<uint32_t>(this);
        break;
    }
    case element::Type_t::f64:
    case element::Type_t::i64:
    case element::Type_t::u64:
    {
        rc = test_bitwise_identical<uint64_t>(this);
        break;
    }
    case element::Type_t::undefined:
    case element::Type_t::dynamic: break;
    }
#pragma GCC diagnostic pop
    return rc;
}

shared_ptr<op::Constant> op::ScalarConstantLikeBase::as_constant() const
{
    return std::make_shared<op::Constant>(m_element_type, m_shape, m_data->get_ptr());
}

std::shared_ptr<Node> op::ScalarConstantLike::copy_with_new_args(const NodeVector& new_args) const
{
    return std::make_shared<ScalarConstantLike>(new_args.at(0), m_value);
}

void op::ScalarConstantLike::infer_element_type()
{
    m_element_type = get_input_element_type(0);
    if (nullptr == m_data)
    {
        m_data.reset(new runtime::AlignedBuffer(m_element_type.size(), m_element_type.size()));
        write_values(std::vector<double>(1, m_value));
    }
}

//
// We have to open up namespace blocks here to work around a problem with gcc:
//
// https://stackoverflow.com/questions/25594644/warning-specialization-of-template-in-different-namespace
//
namespace ngraph
{
    namespace op
    {
        template <>
        void Constant::write_to_buffer<string>(const element::Type& target_type,
                                               const Shape& target_shape,
                                               const vector<string>& source,
                                               void* target,
                                               size_t target_element_count)
        {
        }
    }
}
