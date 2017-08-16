#pragma once

#include <cstdio>
#include <initializer_list>
#include <vector>

#include "element_type.hpp"
#include "tree.hpp"

namespace ngraph
{
    class tensor_size;
    class tensor_stride;
}

//================================================================================================
//
//================================================================================================
class ngraph::tensor_size
{
    friend class tensor_stride;

public:
    tensor_size();
    tensor_size(size_t s, ElementType et = element_type_float);
    tensor_size(const std::initializer_list<scalar_tree>& list,
                ElementType                               et = element_type_float);

    const ElementType& get_type() const { return m_element_type; }
    tensor_stride      full_strides() const;
    tensor_stride      strides() const;
    tensor_size        sizes() const;

    tensor_size operator[](size_t index) const;

    friend std::ostream& operator<<(std::ostream& out, const tensor_size& s);

private:
    tensor_size(const std::vector<size_t>&, const ElementType&);

    scalar_tree m_tree;
    ElementType m_element_type;
};

//================================================================================================
//
//================================================================================================
class ngraph::tensor_stride
{
    friend class tensor_size;

public:
    tensor_stride();
    const ElementType& get_type() const { return m_element_type; }
    tensor_stride      full_strides() const;
    tensor_stride      strides() const;

    tensor_stride reduce_strides() const;

    tensor_stride operator[](size_t index) const;

    friend std::ostream& operator<<(std::ostream& out, const tensor_stride& s);

private:
    tensor_stride(const tensor_size&);
    tensor_stride(const std::vector<size_t>&, const ElementType&);

    scalar_tree m_tree;
    ElementType m_element_type;
};
