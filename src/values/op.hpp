#pragma once

#include <memory>

#include "values/descriptor.hpp"
#include "values/type.hpp"

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

    Broadcast(const Op::ptr_t& x, std::vector<size_t> dims)
    : Op({x}, 0)
    , m_dims(dims)
    {}

public:
    static ptr_t make(const Op::ptr_t& x, std::vector<size_t> dims){
        return ptr_t::make_shared(x, dims);
    }

protected:
    std::vector<size_t> m_dims;
};

class Tuple : public Op
{
public:
    Tuple(const std::vector<ptr_t>& inputs)
    : Op(inputs, 0)
    {
    }
};

} // end of namespace ngraph