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

#include <iostream>
#include <stdexcept>

#include <ngraph/ngraph.hpp>

// Make a runtime tensor for a node output
std::shared_ptr<ngraph::runtime::TensorView> make_output_tensor(
    const std::shared_ptr<ngraph::runtime::Backend>& backend,
    const std::shared_ptr<ngraph::Node>& node,
    size_t output_pos)
{
    return backend->create_tensor(
        node->get_output_element_type(output_pos),
        node->get_output_shape(output_pos));
}

// Initialize a tensor from a random generator
template <typename T>
void randomize(std::function<T()> rand,
               const std::shared_ptr<ngraph::runtime::TensorView>& t)
{
    if (t->get_tensor().get_element_type().bitwidth() != 8 * sizeof(T))
    {
        throw std::invalid_argument(
            "Randomize generator size is not the same as tensor "
            "element size");
    }
    size_t element_count = t->get_element_count();
    std::vector<T> temp;
    for (size_t i = 0; i < element_count; ++i)
    {
        temp.push_back(rand());
    }
    t->write(&temp[0], 0, element_count * sizeof(T));
}

// Get a scalar value from a tensor, optionally at an element offset
template <typename T>
T get_scalar(const std::shared_ptr<ngraph::runtime::TensorView>& t,
             size_t element_offset = 0)
{
    T result;
    t->read(&result, element_offset * sizeof(T), sizeof(T));
    return result;
}

// Set a scalar value in a tensor, optionally at an element offset
template <typename T>
void set_scalar(const std::shared_ptr<ngraph::runtime::TensorView>& t,
                T value,
                size_t element_offset = 0)
{
    t->write(&value, element_offset * sizeof(T), sizeof(T));
}

// Show a shape
std::ostream& operator<<(std::ostream& s, const ngraph::Shape& shape)
{
    s << "Shape{";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        s << shape.at(i);
        if (i + 1 < shape.size())
        {
            s << ", ";
        }
    }
    s << "}";
    return s;
}

// A debug class that supports various ways to dump information about a tensor.
class TensorDumper
{
protected:
    TensorDumper(
        const std::string& name,
        const std::shared_ptr<ngraph::runtime::TensorView>& tensor)
        : m_name(name)
        , m_tensor(tensor)
    {
    }

public:
    virtual ~TensorDumper() {}
    const std::string& get_name() const { return m_name; }
    std::shared_ptr<ngraph::runtime::TensorView> get_tensor() const
    {
        return m_tensor;
    }
    virtual std::ostream& dump(std::ostream& s) const = 0;

protected:
    std::string m_name;
    std::shared_ptr<ngraph::runtime::TensorView> m_tensor;
};

std::ostream& operator<<(std::ostream& s, const TensorDumper& td)
{
    return td.dump(s);
}

// Show the min and max values of a tensor
class MinMax : public TensorDumper
{
public:
    MinMax(const std::string& name,
           const std::shared_ptr<ngraph::runtime::TensorView>& tensor)
        : TensorDumper(name, tensor)
    {
        size_t n = m_tensor->get_element_count();
        for (size_t i = 0; i < n; ++i)
        {
            float s = get_scalar<float>(m_tensor, i);
            m_max = std::max(m_max, s);
            m_min = std::min(m_min, s);
        }
    }

    float get_min() const { return m_min; }
    float get_max() const { return m_max; }
    std::ostream& dump(std::ostream& s) const override
    {
        return s << "MinMax[" << get_name() << ":" << get_min() << ", "
                 << get_max() << "]";
    }

protected:
    float m_min{std::numeric_limits<float>::max()};
    float m_max{std::numeric_limits<float>::min()};
};

// Show the elements of a tensor
class DumpTensor : public TensorDumper
{
public:
    DumpTensor(const std::string& name,
               const std::shared_ptr<ngraph::runtime::TensorView>& tensor)
        : TensorDumper(name, tensor)
    {
    }

    std::ostream& dump(std::ostream& s) const override
    {
        std::shared_ptr<ngraph::runtime::TensorView> t{get_tensor()};
        const ngraph::Shape& shape = t->get_shape();
        s << "Tensor<" << get_name() << ": ";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            s << shape.at(i);
            if (i + 1 < shape.size())
            {
                s << ", ";
            }
        }
        size_t pos = 0;
        s << ">{";
        size_t rank = shape.size();
        if (rank == 0)
        {
            s << get_scalar<float>(t, pos++);
        }
        else if (rank <= 2)
        {
            s << "[";
            for (size_t i = 0; i < shape.at(0); ++i)
            {
                if (rank == 1)
                {
                    s << get_scalar<float>(t, pos++);
                }
                else if (rank == 2)
                {
                    s << "[";
                    for (size_t j = 0; j < shape.at(1); ++j)
                    {
                        s << get_scalar<float>(t, pos++);

                        if (j + 1 < shape.at(1))
                        {
                            s << ", ";
                        }
                    }
                    s << "]";
                }
                if (i + 1 < shape.at(0))
                {
                    s << ", ";
                }
            }
            s << "]";
        }
        s << "}";
        return s;
    }
};
