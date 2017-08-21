#include "values/function.hpp"

using namespace std;
using namespace ngraph;

Parameter::ptr_t Parameter::make(Function& function, size_t index, const ValueType::ptr_t& output_type){
    return ptr_t::make_shared(function, index, output_type);
}

Function::ptr_t Function::make(const ValueType::ptr_t& return_type, const std::vector<ValueType::ptr_t>& argument_types){
    return ptr_t::make_shared(return_type, argument_types);
}

