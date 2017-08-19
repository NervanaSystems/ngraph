#pragma once

#include "values/descriptors.hpp"
#include "values/op.hpp"
#include "values/types.hpp"

namespace ngraph {

class Function;

class Parameter : public Op
{
public:
    using ptr_t = std::shared_ptr<Parameter>;

    static ptr_t make(Function& function, size_t index, const ValueType::ptr_t& output_type);
protected:
    Parameter(Function& function, size_t index, const ValueType::ptr_t& output_type)
    : Op({}, output_type)
    , m_function(function)
    , m_index(index)
    {}

    Function& m_function;
    size_t m_index;
};

class Function
{
public:
    using ptr_t = std::shared_ptr<Function>;

protected:
    Function(const ValueType::ptr_t& return_type, 
             const std::vector<ValueType::ptr_t>& argument_types)
    : m_return_type(return_type)
    , m_argument_types(argument_types)
    {
        size_t i = 0;
        for (auto argument_type : argument_types){
            m_parameters.push_back(Parameter::make(*this, i++, argument_type));
        }
    }

public:
    static ptr_t make(const ValueType::ptr_t& return_type, 
                      const std::vector<ValueType::ptr_t>& argument_types);

    Parameter::ptr_t parameter(size_t i){
        return m_parameters[i];
    }

protected:
    std::vector<Parameter::ptr_t> m_parameters;
    std::vector<std::shared_ptr<ValueType>> m_argument_types;
    std::shared_ptr<ValueType> m_return_type;
};


} // end namespace ngraph