#pragma once

#include <memory>
#include <vector>

#include "element_type.hpp"

namespace ngraph {

class TensorViewDescriptor;
class TupleDescriptor;

using value_size_t = size_t;

class Shape
{
public:
    Shape(const std::initializer_list<value_size_t>& sizes)
    : m_sizes(sizes)
    {}

protected:
    std::vector<value_size_t> m_sizes;
};

// Base type for ngraph values
class ValueType
{
public:
    using ptr_t = std::shared_ptr<ValueType>;
};

class TensorViewType : public ValueType
{
public:
    using ptr_t = std::shared_ptr<TensorViewType>;
    using descriptor_t = TensorViewDescriptor;

    TensorViewType(const ElementType& element_type, const Shape& shape)
    : m_element_type(element_type)
    , m_shape(shape)
    {}

    static ptr_t make(const ElementType& element_type, const Shape& shape){
        return ptr_t::make_shared(element_type, shape);
    }

protected:
    TensorViewType(const TensorViewType&) = delete;
    const ElementType& m_element_type;
    Shape m_shape;
};

class TupleType : public ValueType
{
public:
    using ptr_t = std::shared_ptr<TupleType>;
    using descriptor_t = TupleDescriptor;

    TupleType(const std::vector<ValueType::ptr_t>& element_types)
    : m_element_types(element_types)
    {}

    static ptr_t make(const std::vector<ValueType::ptr_t>& element_types){
        return ptr_t::make_shared(element_types);
    }

protected:
    // Is this name too similar to TensorViewType.to m_element_type?
    std::vector<ValueType::ptr_t> m_element_types;
};

} // End of ngraph