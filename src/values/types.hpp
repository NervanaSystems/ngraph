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
protected:
    ElementType m_element_type;
    std::vector<value_size_t> m_shape;
};

class TupleType : public ValueType
{
protected:
    // Is this name too similar to TensorViewType.to m_element_type?
    std::vector<ValueType> m_element_types;
};

} // End of ngraph