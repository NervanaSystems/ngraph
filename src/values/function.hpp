#pragma once

#include "values/descriptors.hpp"
#include "values/types.hpp"

namespace ngraph {

class Function
{
public:
    using ptr_t = std::shared_ptr<Function>;

    Function(const ValueType::ptr_t& return_type, const std::vector<ValueType::ptr_t>& argument_types)
    : m_return_type(return_type)
    , m_argument_types(argument_types)
    {}

    static ptr_t make(const ValueType::ptr_t& return_type, const std::vector<ValueType::ptr_t>& argument_types){
        return ptr_t(new Function(return_type, argument_types));
    }

protected:
    std::vector<std::shared_ptr<ValueType>> m_argument_types;
    std::shared_ptr<ValueType> m_return_type;
};


} // end namespace ngraph