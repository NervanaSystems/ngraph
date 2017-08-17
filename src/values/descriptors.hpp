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
protected:
    ElementType element_type;
};

class TensorLayoutDescriptor
{

};

class TensorViewDescriptor : public ValueDescriptor
{
public:
    std::shared_ptr<ValueType> value_type() const override {
        return m_type;
    }
protected:
    std::shared_ptr<TensorViewType> m_type;
    TensorDescriptor m_tensor_descriptor;
    TensorLayoutDescriptor m_tensor_layout_descriptor;
};

class TupleDescriptor : public ValueDescriptor
{
public:
    std::shared_ptr<ValueType> value_type() const override {
        return m_type;
    }
protected:
    std::shared_ptr<TupleType> m_type;
    std::vector<ValueDescriptor> m_element_descriptors;
};

} // End of NGRAPH
