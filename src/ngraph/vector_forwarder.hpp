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

#include <initializer_list>
#include <vector>

/// \brief Implements most std::vector behavior for a class by method to an actual vector.
///
/// The methods currently used in ngraph are included. Others may be added as needed.
template <typename ELT_TYPE, typename VECTOR_TYPE>
class VectorForwarder
{
    using vector_t = std::vector<ELT_TYPE>;

public:
    using const_reference = typename vector_t::const_reference;
    using const_iterator = typename vector_t::const_iterator;
    using const_pointer = typename vector_t::const_pointer;
    using const_reverse_iterator = typename vector_t::const_reverse_iterator;
    using difference_type = typename vector_t::difference_type;
    using iterator = typename vector_t::iterator;
    using pointer = typename vector_t::pointer;
    using size_type = typename vector_t::size_type;
    using reference = typename vector_t::reference;
    using reverse_iterator = typename vector_t::reverse_iterator;
    using value_type = typename vector_t::value_type;

    const vector_t& get_vector() const { return m_vector; }
    reference at(size_type n) { return m_vector.at(n); }
    const_reference at(size_type n) const { return m_vector.at(n); }
    iterator begin() { return m_vector.begin(); }
    const_iterator begin() const { return m_vector.begin(); }
    void clear() { m_vector.clear(); }
    bool empty() const { return m_vector.empty(); }
    iterator end() { return m_vector.end(); }
    const_iterator end() const { return m_vector.end(); }
    iterator erase(iterator position) { return m_vector.erase(position); }
    iterator erase(iterator first, iterator last) { return m_vector.erase(first, last); }
    iterator insert(iterator position, const value_type& val)
    {
        return m_vector.insert(position, val);
    }
    void insert(iterator position, size_type n, const value_type& val)
    {
        m_vector.insert(position, n, val);
    }

    template <class InputIterator>
    void insert(iterator position, InputIterator first, InputIterator last)
    {
        m_vector.insert(position, first, last);
    }

    void push_back(const value_type& val) { m_vector.push_back(val); }
    reverse_iterator rbegin() { return m_vector.rbegin(); }
    const_reverse_iterator rbegin() const { return m_vector.rbegin(); }
    reverse_iterator rend() { return m_vector.rend(); }
    const_reverse_iterator rend() const { return m_vector.rend(); }
    size_type size() const { return m_vector.size(); }
    reference operator[](size_type n) { return m_vector[n]; }
    const_reference operator[](size_type n) const { return m_vector[n]; }
    bool operator==(const VECTOR_TYPE& vt) const { return vt.m_vector == m_vector; }
    bool operator==(const vector_t& vt) const { return vt == m_vector; }
    bool operator!=(const VECTOR_TYPE& vt) const { return vt.m_vector != m_vector; }
    bool operator!=(const vector_t& vt) const { return vt != m_vector; }
protected:
    VectorForwarder() {}
    VectorForwarder(size_t n)
        : m_vector(n)
    {
    }
    VectorForwarder(size_t n, value_type v)
        : m_vector(n, v)
    {
    }

    VectorForwarder(const vector_t& v)
        : m_vector(v)
    {
    }

    VectorForwarder(const VECTOR_TYPE& v)
        : m_vector(v.m_vector)
    {
    }

    VectorForwarder(const std::initializer_list<ELT_TYPE>& v)
        : m_vector(v)
    {
    }

protected:
    vector_t m_vector;
};
