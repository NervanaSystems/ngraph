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

// Based on the Matrix class in
// The C++ Programming Language
// Fourth edition
// Bjarne Stroustrup
// Addison-Wesley, Boston, 2013.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "ngraph/log.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace test
    {
        namespace init
        {
            // Recursively define types for N-deep initializer lists
            template <typename T, size_t N>
            struct NestedInitializerListWrapper
            {
                using type =
                    std::initializer_list<typename NestedInitializerListWrapper<T, N - 1>::type>;
            };

            // 1-deep is a plain initializer_list
            template <typename T>
            struct NestedInitializerListWrapper<T, 1>
            {
                using type = std::initializer_list<T>;
            };

            // Scalar case is just the element type
            template <typename T>
            struct NestedInitializerListWrapper<T, 0>
            {
                using type = T;
            };

            // Convenience type name for N-deep initializer lists of Ts
            template <typename T, size_t N>
            using NestedInitializerList = typename NestedInitializerListWrapper<T, N>::type;

            // Fill in a shape from a nested initializer list
            // For a scalar, nothing to do.
            template <typename T, size_t N>
            typename std::enable_if<(N == 0), void>::type
                fill_shape(Shape& shape, const NestedInitializerList<T, N>& inits)
            {
            }

            // Check that the inits match the shape
            template <typename T, size_t N>
            typename std::enable_if<(N == 0), void>::type
                check_shape(const Shape& shape, const NestedInitializerList<T, N>& inits)
            {
                assert(shape.size() == 0);
            }

            // For a plain initializer list, the shape is the length of the list.
            template <typename T, size_t N>
            typename std::enable_if<(N == 1)>::type
                fill_shape(Shape& shape, const NestedInitializerList<T, N>& inits)
            {
                shape.push_back(inits.size());
            }

            template <typename T, size_t N>
            typename std::enable_if<(N == 1)>::type
                check_shape(const Shape& shape, const NestedInitializerList<T, N>& inits)
            {
                assert(shape.at(shape.size() - N) == inits.size());
            }

            // In the general case, we append our level's length and recurse.
            template <typename T, size_t N>
            typename std::enable_if<(N > 1), void>::type
                fill_shape(Shape& shape, const NestedInitializerList<T, N>& inits)
            {
                shape.push_back(inits.size());
                fill_shape<T, N - 1>(shape, *inits.begin());
            }

            template <typename T, size_t N>
            typename std::enable_if<(N > 1), void>::type
                check_shape(const Shape& shape, const NestedInitializerList<T, N>& inits)
            {
                assert(shape.at(shape.size() - N) == inits.size());
                for (auto it : inits)
                {
                    check_shape<T, N - 1>(shape, it);
                }
            }

            // Get the shape of inits.
            template <typename T, size_t N>
            Shape get_shape(const NestedInitializerList<T, N>& inits)
            {
                Shape shape;
                fill_shape<T, N>(shape, inits);
                check_shape<T, N>(shape, inits);
                return shape;
            }

            template <typename IT, typename T, size_t N>
            typename std::enable_if<(N == 1), IT>::type
                flatten(IT it, const Shape& shape, const NestedInitializerList<T, N>& inits)
            {
                assert(inits.size() == shape.at(shape.size() - N));
                for (auto it1 : inits)
                {
                    *(it++) = it1;
                }
                return it;
            }

            template <typename IT, typename T, size_t N>
            typename std::enable_if<(N > 1), IT>::type
                flatten(IT it, const Shape& shape, const NestedInitializerList<T, N>& inits)
            {
                assert(inits.size() == shape.at(shape.size() - N));
                for (auto it1 : inits)
                {
                    it = flatten<IT, T, N - 1>(it, shape, it1);
                }
                return it;
            }

            template <typename IT, typename T, size_t N>
            typename std::enable_if<(N == 0), IT>::type
                flatten(IT it, const Shape& shape, const NestedInitializerList<T, 0>& init)
            {
                assert(shape.size() == 0);
                *(it++) = init;
                return it;
            }
        }

        template <typename T>
        class NDArrayBase
        {
            using vtype = std::vector<T>;

        public:
            using type = T;
            using iterator = typename vtype::iterator;
            using const_iterator = typename vtype::const_iterator;

            NDArrayBase(const Shape& shape)
                : m_shape(shape)
                , m_elements(shape_size(m_shape))
            {
            }

            const Shape& get_shape() const { return m_shape; }
            const_iterator begin() const { return m_elements.begin(); }
            const_iterator end() const { return m_elements.end(); }
            vtype get_vector() { return m_elements; }
            const vtype get_vector() const { return m_elements; }
            operator const vtype() const { return m_elements; }
            operator vtype() { return m_elements; }
            void* data() { return m_elements.data(); }
            const void* data() const { return m_elements.data(); }
            bool operator==(const NDArrayBase<T>& other) const
            {
                return m_shape == other.m_shape && m_elements == other.m_elements;
            }

        protected:
            Shape m_shape;
            vtype m_elements;
        };

        /// An N dimensional array of elements of type T
        template <typename T, size_t N>
        class NDArray : public NDArrayBase<T>
        {
        public:
            NDArray(const init::NestedInitializerList<T, N>& initial_value)
                : NDArrayBase<T>(init::get_shape<T, N>(initial_value))
            {
                init::flatten<typename std::vector<T>::iterator, T, N>(
                    NDArrayBase<T>::m_elements.begin(), NDArrayBase<T>::m_shape, initial_value);
            }
        };
    }
}
