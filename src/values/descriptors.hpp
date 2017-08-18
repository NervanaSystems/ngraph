#pragma once

#include <memory>
#include <vector>

#include "values/types.hpp"

namespace ngraph {

class ValueDescriptor
{
public:
    virtual std::shared_ptr<ValueType> value_type() const = 0;
};

class TensorDescriptor
{
public:
    TensorDescriptor(const ElementType& element_type)
    : m_element_type(element_type)
    {}

protected:
    const ElementType& m_element_type;
};

class TensorLayoutDescriptor
{

};

class TensorViewDescriptor : public ValueDescriptor
{
public:
    TensorViewDescriptor(const std::shared_ptr<TensorViewType>& type)
    : m_type(type)
    {}

    TensorViewDescriptor(const ElementType& element_type, const std::vector<value_size_t>& shape)
    : TensorViewDescriptor(TensorViewType::make_shared(element_type, shape))
    {}

    static std::shared_ptr<TensorViewDescriptor> make_shared(const std::shared_ptr<TensorViewType>& type){
        return std::shared_ptr<TensorViewDescriptor>(new TensorViewDescriptor(type));
    }

    static std::shared_ptr<TensorViewDescriptor> make_shared(const ElementType& element_type, const std::vector<value_size_t>& shape){
        return std::shared_ptr<TensorViewDescriptor>(new TensorViewDescriptor(element_type, shape));
    }

    std::shared_ptr<ValueType> value_type() const override {
        return m_type;
    }
protected:
    std::shared_ptr<TensorViewType> m_type;
    std::shared_ptr<TensorDescriptor> m_tensor_descriptor;
    std::shared_ptr<TensorLayoutDescriptor> m_tensor_layout_descriptor;
};

class TupleDescriptor : public ValueDescriptor
{
public:
    TupleDescriptor(const std::vector<std::shared_ptr<ValueDescriptor>>& elements)
    : m_element_descriptors(elements)
    {
        std::vector<std::shared_ptr<ValueType>> types;
        for(auto elt : elements){
            types.push_back(elt->value_type());
        }
        m_type = TupleType::make_shared(types);
    }

    static std::shared_ptr<TupleDescriptor> make_shared(const std::vector<std::shared_ptr<ValueDescriptor>>& elements){
        return std::shared_ptr<TupleDescriptor>(new TupleDescriptor(elements));
    }

    std::shared_ptr<ValueType> value_type() const override {
        return m_type;
    }
protected:
    std::shared_ptr<TupleType> m_type;
    std::vector<std::shared_ptr<ValueDescriptor>> m_element_descriptors;
};

} // End of NGRAPH
