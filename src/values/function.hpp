#pragma once

#include "values/descriptors.hpp"
#include "values/types.hpp"

namespace ngraph {

class Function
{
    std::vector<std::shared_ptr<ValueDescriptor>> m_arguments;
    std::shared_ptr<ValueDescriptor> m_result;
};


} // end namespace ngraph