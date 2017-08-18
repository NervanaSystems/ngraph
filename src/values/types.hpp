#pragma once

#include <memory>
#include <vector>

#include "element_type.hpp"

namespace ngraph {

using value_size_t = size_t;

// Base type for ngraph values
class ValueType
{

};

class TensorViewType : public ValueType
{
public:
    TensorViewType(const ElementType& element_type, const std::vector<value_size_t>& shape)
    : m_element_type(element_type)
    , m_shape(shape)
    {}

    static std::shared_ptr<TensorViewType> make_shared(const ElementType& element_type, const std::vector<value_size_t>& shape){
        return std::shared_ptr<TensorViewType>(new TensorViewType(element_type, shape));
    }
protected:
    TensorViewType(const TensorViewType&) = delete;
    const ElementType& m_element_type;
    std::vector<value_size_t> m_shape;
};

class TupleType : public ValueType
{
public:
    TupleType(const std::vector<std::shared_ptr<ValueType>>& element_types)
    : m_element_types(element_types)
    {}

    static std::shared_ptr<TupleType> make_shared(const std::vector<std::shared_ptr<ValueType>>& element_types){
        return std::shared_ptr<TupleType>(new TupleType(element_types));
    }

protected:
    // Is this name too similar to TensorViewType.to m_element_type?
    std::vector<std::shared_ptr<ValueType>> m_element_types;
};

} // End of ngraph