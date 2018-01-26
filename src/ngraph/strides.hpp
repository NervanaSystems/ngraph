// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include "vector_forwarder.hpp"

namespace ngraph
{
    /// \brief Strides for a tensor.
    class Strides : public VectorForwarder<size_t, Strides>
    {
    public:
        Strides(const std::initializer_list<size_t>& axes)
            : VectorForwarder<size_t, Strides>(axes)
        {
        }

        Strides(const std::vector<size_t>& axes)
            : VectorForwarder<size_t, Strides>(axes)
        {
        }

        Strides(const Strides& axes)
            : VectorForwarder<size_t, Strides>(axes)
        {
        }

        explicit Strides(size_t n, size_t initial_value = 0)
            : VectorForwarder<size_t, Strides>(n, initial_value)
        {
        }

        Strides() {}
        Strides& operator=(const Strides& v)
        {
            m_vector = v.m_vector;
            return *this;
        }
        Strides& operator=(Strides&& v)
        {
            m_vector = v.m_vector;
            return *this;
        }
    };
}
