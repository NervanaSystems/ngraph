#pragma once

#include <memory>
#include <vector>

#include "element_type.hpp"

namespace ngraph {

using value_size_t = size_t;

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

    TensorViewType(const ElementType& element_type, const std::vector<value_size_t>& shape)
    : m_element_type(element_type)
    , m_shape(shape)
    {}

    static ptr_t make(const ElementType& element_type, const std::vector<value_size_t>& shape){
        return ptr_t(new TensorViewType(element_type, shape));
    }
protected:
    TensorViewType(const TensorViewType&) = delete;
    const ElementType& m_element_type;
    std::vector<value_size_t> m_shape;
};

class TupleType : public ValueType
{
public:
    using ptr_t = std::shared_ptr<TupleType>;

    TupleType(const std::vector<ValueType::ptr_t>& element_types)
    : m_element_types(element_types)
    {}

    static ptr_t make(const std::vector<ValueType::ptr_t>& element_types){
        return ptr_t(new TupleType(element_types));
    }

protected:
    // Is this name too similar to TensorViewType.to m_element_type?
    std::vector<ValueType::ptr_t> m_element_types;
};

} // End of ngraph