/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <algorithm>
#include <vector>

#include "ngraph/axis_set.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    /// \brief Coordinates for a tensor element
    class Coordinate : public std::vector<size_t>
    {
    public:
        Coordinate() {}
        Coordinate(const std::initializer_list<size_t>& axes)
            : std::vector<size_t>(axes)
        {
        }

        Coordinate(const Shape& shape)
            : std::vector<size_t>(static_cast<const std::vector<size_t>&>(shape))
        {
        }

        Coordinate(const std::vector<size_t>& axes)
            : std::vector<size_t>(axes)
        {
        }

        Coordinate(const Coordinate& axes)
            : std::vector<size_t>(axes)
        {
        }

        Coordinate(size_t n, size_t initial_value = 0)
            : std::vector<size_t>(n, initial_value)
        {
        }

        template <class InputIterator>
        Coordinate(InputIterator first, InputIterator last)
            : std::vector<size_t>(first, last)
        {
        }

        Coordinate& operator=(const Coordinate& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }

        Coordinate& operator=(Coordinate&& v)
        {
            static_cast<std::vector<size_t>*>(this)->operator=(v);
            return *this;
        }
    };

    // Removes some values from a vector of axis values
    template <typename AXIS_VALUES>
    AXIS_VALUES project(const AXIS_VALUES& axis_values, const AxisSet& deleted_axes)
    {
        AXIS_VALUES result;

        for (size_t i = 0; i < axis_values.size(); i++)
        {
            if (deleted_axes.find(i) == deleted_axes.end())
            {
                result.push_back(axis_values[i]);
            }
        }

        return result;
    }

    // TODO: check validity, i.e. that the new axis indices are all < axis_values.size()+num_new_axes.
    // Add new values at particular axis positions
    template <typename AXIS_VALUES>
    AXIS_VALUES inject_pairs(const AXIS_VALUES& axis_values,
                             std::vector<std::pair<size_t, size_t>> new_axis_pos_value_pairs)
    {
        AXIS_VALUES result;

        size_t original_pos = 0;

        for (size_t result_pos = 0;
             result_pos < axis_values.size() + new_axis_pos_value_pairs.size();
             result_pos++)
        {
            auto search_it = std::find_if(
                new_axis_pos_value_pairs.begin(),
                new_axis_pos_value_pairs.end(),
                [result_pos](std::pair<size_t, size_t> p) { return p.first == result_pos; });

            if (search_it == new_axis_pos_value_pairs.end())
            {
                result.push_back(axis_values[original_pos++]);
            }
            else
            {
                result.push_back(search_it->second);
            }
        }

        return result;
    }

    // Add a new value at a particular axis position
    template <typename AXIS_VALUES>
    AXIS_VALUES inject(const AXIS_VALUES& axis_values, size_t new_axis_pos, size_t new_axis_val)
    {
        return inject_pairs(axis_values,
                            std::vector<std::pair<size_t, size_t>>{
                                std::pair<size_t, size_t>(new_axis_pos, new_axis_val)});
    }

    std::ostream& operator<<(std::ostream& s, const Coordinate& coordinate);
}
