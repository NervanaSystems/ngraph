#pragma once

#include <memory>

#include "values/descriptors.hpp"
#include "values/types.hpp"

namespace ngraph {

class Op
{
public:
    using ptr_t = std::shared_ptr<Op>;

protected:
    Op(const std::vector<ptr_t>& inputs, const ValueType::ptr_t output_type)
    : m_inputs(inputs)
    , m_output_type(output_type)
    {}

    std::vector<ptr_t> m_inputs;
    ValueType::ptr_t m_output_type;
};

class Broadcast : public Op
{
public:
    using ptr_t = std::shared_ptr<Broadcast>;

protected:
    Broadcast(const Op::ptr_t& x, std::vector<size_t> dims)
    : Op({x}, 0)
    , m_dims(dims)
    {}

public:
    static ptr_t make(const Op::ptr_t& x, std::vector<size_t> dims){
        return ptr_t(new Broadcast(x, dims));
    }

protected:
    std::vector<size_t> m_dims;
};

} // end of namespace ngraph