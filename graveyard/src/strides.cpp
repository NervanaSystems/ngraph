#include <algorithm>
#include <iostream>

#include "strides.hpp"
#include "util.hpp"

using namespace std;

//================================================================================================
//
//================================================================================================

ngraph::tensor_size::tensor_size()
    : m_tree{}
    , m_element_type{element_type_float}
{
}

ngraph::tensor_size::tensor_size(size_t s, ElementType et)
    : m_tree{s}
    , m_element_type{et}
{
}

ngraph::tensor_size::tensor_size(const std::initializer_list<scalar_tree>& list, ElementType et)
    : m_tree{list}
    , m_element_type{et}
{
}

ngraph::tensor_size::tensor_size(const std::vector<size_t>& list, const ElementType& et)
    : m_tree{list}
    , m_element_type{et}
{
}

ngraph::tensor_stride ngraph::tensor_size::full_strides() const
{
    tensor_stride   result{*this};
    vector<size_t*> value_pointer_list;
    vector<size_t>  size_list;

    scalar_tree::traverse_tree(result.m_tree, [&](size_t* value) {
        value_pointer_list.push_back(value);
        size_list.push_back(*value);
    });
    int index                  = value_pointer_list.size() - 1;
    *value_pointer_list[index] = result.m_element_type.size();
    for (index--; index >= 0; index--)
    {
        *value_pointer_list[index] = *value_pointer_list[index + 1] * size_list[index + 1];
    }

    return result;
}

ngraph::tensor_stride ngraph::tensor_size::strides() const
{
    return full_strides().strides();
}

ngraph::tensor_size ngraph::tensor_size::sizes() const
{
    vector<size_t> tmp;
    if (m_tree.is_list())
    {
        for (auto s : m_tree.get_list())
        {
            tmp.push_back(s.reduce([](size_t a, size_t b) { return a * b; }));
        }
    }
    else
    {
        tmp.push_back(m_tree.get_value());
    }
    return tensor_size(tmp, m_element_type);
}

std::ostream& ngraph::operator<<(std::ostream& out, const ngraph::tensor_size& s)
{
    out << s.m_tree;
    return out;
}

//================================================================================================
//
//================================================================================================

ngraph::tensor_stride::tensor_stride()
    : m_tree{}
    , m_element_type{element_type_float}
{
}

ngraph::tensor_stride::tensor_stride(const tensor_size& s)
    : m_tree{}
    , m_element_type{s.m_element_type}
{
    m_tree = s.m_tree;
}

ngraph::tensor_stride::tensor_stride(const std::vector<size_t>& list, const ElementType& et)
    : m_tree{}
    , m_element_type{et}
{
    m_tree = list;
}

ngraph::tensor_stride ngraph::tensor_stride::reduce_strides() const
{
    vector<size_t> tmp;
    if (m_tree.is_list())
    {
        for (auto s : m_tree.get_list())
        {
            tmp.push_back(s.reduce([](size_t a, size_t b) { return min(a, b); }));
        }
    }
    else
    {
        tmp.push_back(m_tree.get_value());
    }
    return tensor_stride(tmp, m_element_type);
}

ngraph::tensor_stride ngraph::tensor_stride::full_strides() const
{
    return *this;
}

ngraph::tensor_stride ngraph::tensor_stride::strides() const
{
    return reduce_strides();
}

std::ostream& ngraph::operator<<(std::ostream& out, const ngraph::tensor_stride& s)
{
    out << s.m_tree;
    return out;
}
