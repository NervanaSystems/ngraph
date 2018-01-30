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
#include <set>

/// \brief Implements most std::set behavior for a class by method to an actual set.
///
/// The methods currently used in ngraph are included. Others may be added as needed.
template <typename ELT_TYPE, typename SET_TYPE>
class SetForwarder
{
    using set_t = std::set<ELT_TYPE>;

public:
    using const_reference = typename set_t::const_reference;
    using const_iterator = typename set_t::const_iterator;
    using const_pointer = typename set_t::const_pointer;
    using const_reverse_iterator = typename set_t::const_reverse_iterator;
    using difference_type = typename set_t::difference_type;
    using iterator = typename set_t::iterator;
    using key_type = typename set_t::key_type;
    using pointer = typename set_t::pointer;
    using size_type = typename set_t::size_type;
    using reference = typename set_t::reference;
    using reverse_iterator = typename set_t::reverse_iterator;
    using value_type = typename set_t::value_type;

    iterator begin() { return m_set.begin(); }
    const_iterator begin() const { return m_set.begin(); }
    void clear() { m_set.clear(); }
    size_type count(const value_type& val) const { return m_set.count(val); }
    iterator end() { return m_set.end(); }
    const_iterator end() const { return m_set.end(); }
    iterator erase(iterator position) { return m_set.erase(position); }
    iterator erase(iterator first, iterator last) { return m_set.erase(first, last); }
    iterator find(const value_type& val) { return m_set.find(val); }
    const_iterator find(const value_type& val) const { return m_set.find(val); }
    std::pair<iterator, bool> insert(const value_type& val) { return m_set.insert(val); }
    std::pair<iterator, bool> insert(value_type&& val) { return m_set.insert(val); }
    iterator insert(const_iterator position, const value_type& val)
    {
        return m_set.insert(position, val);
    }
    iterator insert(const_iterator position, value_type&& val)
    {
        return m_set.insert(position, val);
    }
    template <class InputIterator>
    void insert(InputIterator first, InputIterator last)
    {
        m_set.insert(first, last);
    }
    void insert(std::initializer_list<value_type> il) { return m_set.insert(il); }
    void push_back(const value_type& val) { m_set.push_back(val); }
    reverse_iterator rbegin() { return m_set.rbegin(); }
    const_reverse_iterator rbegin() const { return m_set.rbegin(); }
    reverse_iterator rend() { return m_set.rend(); }
    const_reverse_iterator rend() const { return m_set.rend(); }
    size_type size() const { return m_set.size(); }
    bool operator==(const SET_TYPE& vt) const { return vt.m_set == m_set; }
    bool operator==(const set_t& vt) const { return vt == m_set; }
    bool operator!=(const SET_TYPE& vt) const { return vt.m_set != m_set; }
    bool operator!=(const set_t& vt) const { return vt != m_set; }
    operator set_t&() { return m_set; }
    operator const set_t&() const { return m_set; }
protected:
    SetForwarder() {}
    SetForwarder(const set_t& v)
        : m_set(v)
    {
    }

    SetForwarder(const SET_TYPE& v)
        : m_set(v.m_set)
    {
    }

    SetForwarder(const std::initializer_list<ELT_TYPE>& v)
        : m_set(v)
    {
    }

protected:
    set_t m_set;
};
