//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cstddef>
#include <iterator>

#include "exceptions.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        template <typename T>
        class Span
        {
        public:
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using value_type = T;
            using reference = value_type&;
            using pointer = value_type*;
            using const_reference = const value_type&;
            using const_pointer = const value_type*;
            using iterator = pointer;
            using const_iterator = const_pointer;
            using reverse_iterator = std::reverse_iterator<iterator>;
            using const_reverse_iterator = const reverse_iterator;

            Span(const Span&) = default;
            Span& operator=(const Span&) = default;

            Span() = default;

            Span(Span&&) noexcept = default;
            Span& operator=(Span&&) noexcept = default;

            template <typename K>
            Span(const K* buffer, size_type count)
                : m_begin{reinterpret_cast<pointer>(const_cast<K*>(buffer))}
                , m_end{reinterpret_cast<pointer>(const_cast<K*>(buffer)) + count}
                , m_count{count}
            {
            }

            iterator begin() { return m_begin; }
            iterator end() { return m_end; }
            const_iterator begin() const { return m_begin; }
            const_iterator end() const { return m_end; }
            const_iterator cbegin() const { return m_begin; }
            const_iterator cend() const { return m_end; }
            reverse_iterator rbegin() { return reverse_iterator{m_end}; }
            const_reverse_iterator crbegin() const { return const_reverse_iterator{m_end}; }
            reverse_iterator rend() { return reverse_iterator{m_begin}; }
            const_reverse_iterator crend() const { return const_reverse_iterator{m_begin}; }
            const_reference at(std::size_t index) const
            {
                auto it = std::next(m_begin, index);
                if (it >= m_end)
                {
                    throw std::out_of_range{"span"};
                }
                return *it;
            }

            reference at(std::size_t index)
            {
                auto it = std::next(m_begin, index);
                if (it >= m_end)
                {
                    throw std::out_of_range{"span"};
                }
                return *it;
            }

            reference front() { return *m_begin; }
            const_reference front() const { return *m_begin; }
            reference back() { return *std::prev(m_end); }
            const_reference back() const { return *std::prev(m_end); }
            const_pointer data() const { return m_begin; }
            reference operator[](std::size_t index) { return at(index); }
            const_reference operator[](std::size_t index) const { return at(index); }
            size_type size() const { return m_count; }
            bool is_valid() const { return (m_begin != nullptr) && (m_count > 0); }
            bool empty() const { return (m_count == 0); }
        private:
            iterator m_begin{nullptr}, m_end{nullptr};
            size_type m_count{0};
        };
    }
}
