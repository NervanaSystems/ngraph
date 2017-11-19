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

namespace ngraph
{
    namespace runtime
    {
        template <typename T>
        class TensorData
        {
        public:
            TensorData(T* data, size_t size)
                : m_data(data)
                , m_size(size)
            {
            }

            class TensorIterator : public std::iterator<std::input_iterator_tag, T>
            {
                T* p;

            public:
                TensorIterator(T* x)
                    : p(x)
                {
                }
                TensorIterator(const TensorIterator& mit)
                    : p(mit.p)
                {
                }
                TensorIterator& operator++()
                {
                    ++p;
                    return *this;
                }
                TensorIterator operator++(int)
                {
                    TensorIterator tmp(*this);
                    operator++();
                    return tmp;
                }
                bool operator==(const TensorIterator& rhs) const { return p == rhs.p; }
                bool operator!=(const TensorIterator& rhs) const { return p != rhs.p; }
                T& operator*() { return *p; }
            };
            TensorIterator begin() { return TensorIterator(m_data); }
            TensorIterator end() { return TensorIterator(&m_data[m_size]); }
        private:
            T* m_data;
            size_t m_size;
        };
    }
}
