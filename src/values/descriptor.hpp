#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "values/type.hpp"

namespace ngraph {

class ValueDescriptor
{
public:
    using ptr_t = std::shared_ptr<ValueDescriptor>;

    virtual ValueType::ptr_t value_type() const = 0;
};

class TensorDescriptor
{
public:
    using ptr_t = std::shared_ptr<TensorDescriptor>;

    TensorDescriptor(const ElementType& element_type)
    : m_element_type(element_type)
    {}

protected:
    const ElementType& m_element_type;
};

class TensorLayoutDescriptor
{
public:
    using ptr_t = std::shared_ptr<TensorLayoutDescriptor>;
};

class TensorViewDescriptor : public ValueDescriptor
{
public:
    using ptr_t = std::shared_ptr<TensorViewDescriptor>;

    TensorViewDescriptor(const TensorViewType::ptr_t& type)
    : m_type(type)
    {}

    TensorViewDescriptor(const ElementType& element_type, const Shape& shape)
    : TensorViewDescriptor(TensorViewType::make(element_type, shape))
    {}

    static ptr_t make(const TensorViewType::ptr_t& type){
        return ptr_t::make_shared(type);
    }

    static ptr_t make(const ElementType& element_type, const Shape& shape){
        return ptr_t::make_shared(element_type, shape);
    }

    ValueType::ptr_t value_type() const override {
        return m_type;
    }
protected:
    TensorViewType::ptr_t m_type;
    TensorDescriptor::ptr_t m_tensor_descriptor;
    TensorLayoutDescriptor::ptr_t m_tensor_layout_descriptor;
};

class TupleDescriptor : public ValueDescriptor
{
public:
    using ptr_t = std::shared_ptr<TupleDescriptor>;

    TupleDescriptor(const std::vector<ValueDescriptor::ptr_t>& elements)
    : m_element_descriptors(elements)
    {
        std::vector<ValueType::ptr_t> types;
        for(auto elt : elements){
            types.push_back(elt->value_type());
        }
        m_type = TupleType::make(types);
    }

    static ptr_t make(const std::vector<ValueDescriptor::ptr_t>& elements){
        return ptr_t::make_shared(elements);
    }

    ValueType::ptr_t value_type() const override {
        return m_type;
    }
protected:
    TupleType::ptr_t m_type;
    std::vector<ValueDescriptor::ptr_t> m_element_descriptors;
};

} // End of NGRAPH
