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

#include <cstddef>

#include "ngraph/vector_forwarder.hpp"

namespace ngraph
{
    /// \brief A vector of axes.
    class AxisVector : public VectorForwarder<size_t, AxisVector>
    {
    public:
        AxisVector(const std::initializer_list<size_t>& axes)
            : VectorForwarder<size_t, AxisVector>(axes)
        {
        }

        AxisVector(const std::vector<size_t>& axes)
            : VectorForwarder<size_t, AxisVector>(axes)
        {
        }

        AxisVector(const AxisVector& axes)
            : VectorForwarder<size_t, AxisVector>(axes)
        {
        }

        explicit AxisVector(size_t n)
            : VectorForwarder<size_t, AxisVector>(n)
        {
        }

        AxisVector() {}
        AxisVector& operator=(const AxisVector& v)
        {
            m_vector = v.m_vector;
            return *this;
        }
        AxisVector& operator=(AxisVector&& v)
        {
            m_vector = v.m_vector;
            return *this;
        }
    };
}
